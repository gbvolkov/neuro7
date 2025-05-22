# yandex_realty_parser.py
#
# A comprehensive parser for Yandex Realty XML feeds (sale housing, commercial, new construction)
# Supports full field set per Yandex specifications:
# https://yandex.ru/support/realty/ru/rules/content-requirements
# https://yandex.ru/support/realty/ru/requirements/requirements-sale-housing
# https://yandex.ru/support/realty/ru/requirements/requirements-commercial
# https://yandex.ru/support/realty/ru/requirements/requirements-sale-new

import sqlite3
from lxml import etree
import sys
import re

# Usage: python yandex_realty_parser.py <input_xml> [<output_db>]
# Defaults: input_xml='vesna.xml', output_db='realty.db'


# XML namespace
NS = {'y': 'http://webmaster.yandex.ru/schemas/feed/realty/2010-06'}

# Helper to extract text or None
def get_text(el, path):
    node = el.find(path, namespaces=NS)
    return node.text.strip() if node is not None and node.text else None

# Safe parsing functions
def parse_int(value):
    if not value:
        return None
    match = re.match(r"-?\d+", value)
    return int(match.group()) if match else None

def parse_float(value):
    if not value:
        return None
    val = value.strip().lower()
    if val in ('null', ''):
        return None
    try:
        return float(val)
    except ValueError:
        return None

# Parse apartment string (e.g. '3(103)') -> int(3)
def parse_apartment(value):
    return parse_int(value)

# Initialize database schema
def init_db(conn):
    c = conn.cursor()
    # Main offers table with full fields
    c.execute('''
    CREATE TABLE IF NOT EXISTS offers (
        internal_id        TEXT PRIMARY KEY,
        type                TEXT,
        category            TEXT,
        property_type       TEXT,
        creation_date       TEXT,
        deal_status         TEXT,
        url                 TEXT,
        price_value         REAL,
        price_currency      TEXT,
        country             TEXT,
        region              TEXT,
        locality_name       TEXT,
        sub_locality_name   TEXT,
        address             TEXT,
        apartment           INTEGER,
        latitude            REAL,
        longitude           REAL,
        rooms               INTEGER,
        area_total          REAL,
        area_live           REAL,
        area_kitchen        REAL,
        area_unit           TEXT,
        floor               INTEGER,
        floors_total        INTEGER,
        building_name       TEXT,
        building_section    TEXT,
        building_state      TEXT,
        built_year          INTEGER,
        ready_quarter       INTEGER,
        renovation          TEXT,
        new_flat            INTEGER,
        sales_agent_id      TEXT,
        elevator            INTEGER,
        parking             TEXT
    )''')
    # Images table
    c.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        offer_id TEXT,
        url      TEXT,
        FOREIGN KEY(offer_id) REFERENCES offers(internal_id)
    )''')
    # Sales agents table
    c.execute('''
    CREATE TABLE IF NOT EXISTS sales_agents (
        agent_id     TEXT PRIMARY KEY,
        name         TEXT,
        organization TEXT,
        category     TEXT,
        phone        TEXT,
        url          TEXT
    )''')
    conn.commit()

# Parse feed and load into DB
def parse_and_load(xml_file, conn):
    tree = etree.parse(xml_file)
    offers = tree.findall('.//y:offer', namespaces=NS)
    c = conn.cursor()

    for offer in offers:
        oid = offer.get('internal-id')
        # Core fields
        core = {
            'internal_id': oid,
            'type': offer.get('type'),
            'category': get_text(offer, 'y:category'),
            'property_type': get_text(offer, 'y:property-type'),
            'creation_date': get_text(offer, 'y:creation-date'),
            'deal_status': get_text(offer, 'y:deal-status'),
            'url': get_text(offer, 'y:url')
        }
        # Location and apartment
        loc = offer.find('y:location', namespaces=NS)
        if loc is not None:
            core |= {
                'country': get_text(loc, 'y:country'),
                'region': get_text(loc, 'y:region'),
                'locality_name': get_text(loc, 'y:locality-name'),
                'sub_locality_name': get_text(loc, 'y:sub-locality-name'),
                'address': get_text(loc, 'y:address'),
                'apartment': parse_apartment(get_text(loc, 'y:apartment')),
                'latitude': parse_float(get_text(loc, 'y:latitude')),
                'longitude': parse_float(get_text(loc, 'y:longitude')),
            }
        # Price
        price = offer.find('y:price', namespaces=NS)
        if price is not None:
            core |= {
                'price_value': parse_float(get_text(price, 'y:value')),
                'price_currency': get_text(price, 'y:currency'),
            }
        # Area breakdown
        core['area_unit'] = get_text(offer, 'y:area/y:unit')
        core['area_total'] = parse_float(get_text(offer, 'y:area/y:value'))
        core['area_live'] = parse_float(get_text(offer, 'y:living-space'))
        core['area_kitchen'] = parse_float(get_text(offer, 'y:kitchen-space'))
        # Structure and building info
        core |= {
            'rooms': parse_int(get_text(offer, 'y:rooms')),
            'floor': parse_int(get_text(offer, 'y:floor')),
            'floors_total': parse_int(get_text(offer, 'y:floors-total')),
            'building_name': get_text(offer, 'y:building-name'),
            'building_section': get_text(offer, 'y:building-section'),
            'building_state': get_text(offer, 'y:building-state'),
            'built_year': parse_int(get_text(offer, 'y:built-year')),
            'ready_quarter': parse_int(get_text(offer, 'y:ready-quarter')),
            'renovation': get_text(offer, 'y:renovation'),
            'new_flat': 1 if get_text(offer, 'y:new-flat') == 'true' else 0,
            'elevator': 1 if get_text(offer, 'y:elevator') == 'yes' else 0,
            'parking': get_text(offer, 'y:parking'),
        }
        # Sales agent
        agent = offer.find('y:sales-agent', namespaces=NS)
        if agent is not None:
            agent_id = f"agent_{oid}"
            core['sales_agent_id'] = agent_id
            agent_data = {
                'agent_id': agent_id,
                'name': get_text(agent, 'y:name'),
                'organization': get_text(agent, 'y:organization'),
                'category': agent.get('category'),
                'phone': get_text(agent, 'y:phone'),
                'url': get_text(agent, 'y:url')
            }
            c.execute(
                "INSERT OR REPLACE INTO sales_agents(agent_id,name,organization,category,phone,url)"
                " VALUES(:agent_id,:name,:organization,:category,:phone,:url)", agent_data
            )
        # Insert or replace offer
        cols = ','.join(core.keys())
        placeholders = ':' + ',:'.join(core.keys())
        c.execute(f"INSERT OR REPLACE INTO offers({cols}) VALUES({placeholders})", core)
        # Images
        for img in offer.findall('y:image', namespaces=NS):
            img_url = img.text.strip() if img.text else None
            if img_url:
                c.execute("INSERT INTO images(offer_id,url) VALUES(?,?)", (oid, img_url))
    conn.commit()

if __name__ == '__main__':
    #XML_FILE, DB_FILE = get_args()

    files = ["data/pricing/7ya", "data/pricing/vesna", "data/pricing/andersen"]
    for file in files:
        XML_FILE = f"{file}.xml"
        DB_FILE = f"{file}.db"
        conn = sqlite3.connect(DB_FILE)
        init_db(conn)
        parse_and_load(XML_FILE, conn)
        conn.close()
        print(f"Loaded {XML_FILE} into {DB_FILE} successfully.")
