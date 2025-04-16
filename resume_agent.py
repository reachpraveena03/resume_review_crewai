import os
import fitz  # PyMuPDF
import docx  # python-docx

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Get API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key = os.environ.get("SERPER_API_KEY")

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Initialize the Serper tool with API key
#search_tool = SerperDevTool(api_key=serper_api_key)

# Resume parsing functions
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    fulltext = []
    for para in doc.paragraphs:
        fulltext.append(para.text)
    return "\n".join(fulltext)

def extract_text_from_resume(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format"

# Read and print resume
resume_path = "C:/Users/User/Downloads/Praveena Lingala-Data Scientist Position.pdf"
res1 = extract_text_from_resume(resume_path)
print(res1)

# Define the agents
resume_feedback = Agent(
    role="Professional Resume Advisor",
    goal="Give feedback on the resume to make it stand out in the job market.",
    backstory="With a strategic mind and an eye for detail, you excel at providing feedback on resumes to highlight the most relevant skills and experiences.",
    verbose=True,
    llm=llm
)

resume_advisor = Agent(
    role="Professional Resume Writer",
    goal="Based on the feedback received, rewrite the resume to highlight the most relevant skills and experience.",
    backstory="With a keen eye for talent presentation, you craft resumes that grab attention and reflect a candidate's true potential.",
    verbose=True,
    llm=llm
)

#job_researcher = Agent(
#     role="Senior Recruitment Consultant",
#     goal="Find the 5 most relevant, recently posted jobs based on the improved resume and location preference.",
#     backstory="As a senior recruitment consultant, your prowess lies in identifying the most suitable job openings using modern search tools.",
#     tools=[search_tool],
#     verbose=True,
#     llm=llm
# )

# Define tasks
resume_feedback_task = Task(
    description=(
        "Give feedback on the resume to make it stand out for recruiters. "
        "Review every section, including the summary, work experience, skills, and education. "
        "Suggest adding relevant sections if they are missing. Also give an overall score to the resume out of 10. "
        "This is the resume: {resume}"
    ),
    expected_output="The overall score of the resume followed by the feedback in bullet points.",
    agent=resume_feedback
)

resume_advisor_task = Task(
    description=(
        "Rewrite the resume based on the feedback to make it stand out for recruiters. "
        "You can adjust and enhance the resume but don't make up facts. Review every section to reflect the candidate's abilities. "
        "This is the resume: {resume}"
    ),
    expected_output="Resume in markdown format that effectively highlights the candidate's qualifications and experiences.",
    context=[resume_feedback_task],
    agent=resume_advisor
)

# research_task = Task(
#     description=(
#         "Find the 5 most relevant job postings based on the resume received from the resume advisor and location preference. "
#         "Use the tool to gather relevant content and shortlist the most recent job openings."
#     ),
#     expected_output=(
#         "A bullet point list of the 5 job openings, with links and detailed descriptions for each job, in markdown format."
#     ),
#     context=[resume_advisor_task],
#     agent=job_researcher
# )

# Assemble the Crew
crew = Crew(
    agents=[resume_feedback, resume_advisor],
    tasks=[resume_feedback_task, resume_advisor_task],
    verbose=True
)

# Run the workflow
result = crew.kickoff(inputs={"resume": res1})
print(result)