import os
from dotenv import load_dotenv, find_dotenv
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner,
    function_tool
)

import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent




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



# ================================ TOOLS ===============================
@function_tool
def get_career_roadmap(career_field: str) -> str:
    """
    Generates a skill roadmap for the selected career field.
    
    Args:
        career_field (str): The name of the career field selected by the user.

    Returns:
        str: A roadmap that outlines the essential skills and steps required to pursue a career in the given field.
    """
    sample_roadmaps = {
        "Data Science": """
        Career Roadmap: Data Science
        1. Learn Python and R
        2. Understand statistics and probability
        3. Get hands-on with libraries like NumPy, Pandas, Scikit-learn
        4. Learn SQL and data visualization tools (e.g., Tableau, Power BI)
        5. Study machine learning concepts
        6. Work on real-world data projects and build a portfolio
        7. Consider certifications or a master's in data science
        """,
        "Web Development": """
        Career Roadmap: Web Development
        1. Learn HTML, CSS, and JavaScript
        2. Master front-end frameworks like React or Vue
        3. Learn back-end development with Node.js, Django, or Flask
        4. Understand databases (SQL, MongoDB)
        5. Practice version control (Git & GitHub)
        6. Build full-stack projects and deploy them
        7. Stay updated with industry trends and tools
        """,
        "UI/UX Design": """
        Career Roadmap: UI/UX Design
        1. Understand design principles and color theory
        2. Learn tools like Figma, Adobe XD, or Sketch
        3. Study user research and persona creation
        4. Practice wireframing and prototyping
        5. Build a design portfolio with case studies
        6. Get feedback and iterate on designs
        7. Apply for internships or freelance gigs
        """,
    }

    # Return the roadmap if available, otherwise a default message
    roadmap = sample_roadmaps.get(career_field.title())
    if roadmap:
        return roadmap
    else:
        return f"""
        Career Roadmap for '{career_field}':
        1. Research the field and required qualifications
        2. Identify top skills in demand using platforms like LinkedIn or Glassdoor
        3. Enroll in relevant courses (Coursera, Udemy, etc.)
        4. Gain hands-on experience through projects or internships
        5. Build a professional portfolio
        6. Network with professionals in the field
        7. Apply for jobs or freelance opportunities
        """




# ============================= SAME STEPS FOR OPENAI & GEMINI ====================================
CareerAgent = Agent(
    name="CareerAgent",
    instructions="""
    Help the user with career-related questions. Suggest suitable career fields based on user interests,
    and explain the present and future scope of those fields. use tool get_career_roadmap to generate 
    roadmap based on user interested career.
    """,
    tools=[get_career_roadmap]
)

SkillAgent = Agent(
    name="SkillAgent",
    instructions="""
    Assist the user in building skill plans based on their career interests. Provide step-by-step guidance
    on learning and developing relevant skills.
    """
)

JobAgent = Agent(
    name="JobAgent",
    instructions="""
    Share information about real-world job roles based on the user's career path and skills.
    Discuss job responsibilities, scope, and growth opportunities.
    """
)

MainAgent = Agent(
    name="Career Mentor Agent",
    instructions="""
    Handle all questions related to careers, academics, jobs, and skill-building.
    - If the user asks about career fields or interests → hand off to CareerAgent.
    - If the user asks about skills or how to build them → hand off to SkillAgent.
    - If the user asks about job roles or opportunities → hand off to JobAgent.
    - If the user asks about the full roadmap from career to job → explain the full path and guide accordingly.
    """,
    handoffs=[CareerAgent, SkillAgent, JobAgent]
)





@cl.on_message
async def main(message: cl.Message):
    try:
        msg = cl.Message(content='Thinking...')
        await msg.send()


        result = Runner.run_streamed(
            MainAgent,
            input = message.content,
            run_config=run_config
        )

        collected = ''

        async for event in result.stream_events():
            if event.type=="raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                collected += token
                await msg.stream_token(token)
        
        msg.content = collected
        await msg.update()

    except Exception as e:
        await cl.Message(str(e)).send()

