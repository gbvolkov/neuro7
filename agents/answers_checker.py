from langchain_openai import ChatOpenAI



check_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def check_summary(answer):
    prompt = "Ты агент, который определяет наличие итоговой записи в сообщении.\n" \
        "Ты получаешь на вход ответ агента по недвижимости Клиенту.\n" \
        "Если ответ содержит a Summary of a chat with client, including agreed with manager call time, information about client, it's family, reason for purchasing flat, building complex, financial conditions client is interested in, number of rooms, budget (if provided, optional), всегда отвечай 'YES'" \
        "В противном случае всегда отвечай 'NO'\n\n" \
        f"AgentAnswer: {answer}."

    result = check_llm.invoke(prompt)
    return "YES" if "YES" in result.content else "NO"

