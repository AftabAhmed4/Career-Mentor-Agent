import os
from dotenv import load_dotenv, find_dotenv
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner
)




load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")



# ======================== GEMINI KEYS STEP START ======================== 
provider = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" ,
    api_key=gemini_api_key,
)


model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)


run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# ======================== GEMINI KEYS STEP END ======================== 


# ============================= SAME STEPS ====================================
agent = Agent(
    name="Career Mentor Agent",
    instructions="Provides help for academics questions"
)


result = Runner.run_sync(
    agent,
    input="What is the capital of pakistan",
    run_config=run_config   # for GEMINI only
)

print(result.final_output)


