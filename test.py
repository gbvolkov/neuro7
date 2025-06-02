from utils.utils import ModelType
from agents.supervisor import initialize_agent
import uuid


if __name__ == "__main__":
    agent = initialize_agent(model=ModelType.GPT)

    #show_graph(assistant_graph)
    from langchain_core.messages import HumanMessage

    # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "В каких ЖК вы предлагаете квартиры?",
        "Какие есть двушки в 7Я?",
        #"А кто строил?",
        #"А какие объекты уже сдали?",
        #"А Александрит ваш?",
        "Я хочу в районе весенней что-нибудь.",
        "Не, не весна-парк, другой в этом районе.",
        "Ну а чё у вас в районе Вессенней?",
        "А Андерсен в каком районе?",
        "А когда сдаёте его?",
        "Расскажи подробнее. У меня сын. Чё там для него есть? Ну вообще - там магазы, чё-нить такое",
        "Подбери мне там квартиру стоимостью до 10 млн и площадью от 40 кв.м. и напиши, какие финансовые усоловия - скидки там и пр",
        "А можешь скинуть список вариантов квартир?",
        "Давай созвонимся сегодня после 17:00",
    ]

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_info": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    #_printed = set()
    for question in tutorial_questions:

        response = agent.invoke({"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config)

        #events = agent.stream(
        #    {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config, stream_mode="values"
        #)
        answer = response['messages'][-1].content
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        print(answer)
        #for event in events:
        #    #_print_event(event, _printed)
        #    _print_response(event, _printed)
        print("===================")

    #print("RESET")
    #events = assistant_graph.invoke(
    #    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, config, stream_mode="values"
    #)

