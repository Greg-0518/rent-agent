from langchain.chat_models import init_chat_model


def getModel(is_thinking: bool = False):
    model_name = None
    if is_thinking:
        model_name = 'deepseek-reasoner'
    else:
        model_name = 'deepseek-chat'

    return init_chat_model(model=model_name, temperature=0)


model = getModel(True)