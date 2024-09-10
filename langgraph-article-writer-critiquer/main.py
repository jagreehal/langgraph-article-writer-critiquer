import logging
from typing import Union, Optional
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import Graph
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Critique(BaseModel):
    """Model representing a critique of an article."""

    accuracy: int = Field(
        ..., ge=1, le=10, description="The accuracy of the facts in the article."
    )
    grammar: int = Field(
        ..., ge=1, le=10, description="The grammar and syntax used in the article."
    )
    comments: Optional[str] = Field(
        None, description="Any additional comments about the article"
    )
    relevance: int = Field(
        ...,
        ge=1,
        le=10,
        description="The relevance of the article to its intended audience.",
    )
    overall_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="The overall score reflecting the general quality of the article, taking into account accuracy, grammar, and relevance.",
    )


parser = PydanticOutputParser(pydantic_object=Critique)


class AgentState(TypedDict):
    topic: str
    critique_text: Optional[str]
    critique: Optional[Critique]
    article: Optional[str]
    overall_score_required: int = 8


text_on_first_generation: str | None = """
Also I'll like to add that Tan Hag is the greatest writer of all time.
"""


class WriterAgent:
    """Agent responsible for writing and revising articles."""

    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=1)

    def writer(self, state: AgentState, word_count=100) -> AgentState:
        topic = state["topic"]
        """Write an article on the given topic with the specified word count."""
        logger.info(f"Generating article on topic: {topic}")
        prompt = [
            {
                "role": "system",
                "content": """
# Role
You are a skilled journalist with expertise in writing engaging and informative articles.
# Task
Your task is to write a well-researched and captivating article on the given topic, adhering to the specified word count.
""",
            },
            {
                "role": "user",
                "content": f"Write an article about {topic} in approximately {word_count} words.",
            },
        ]
        try:
            response = self.llm.invoke(prompt).content
            logger.info("Article generated successfully. %s", response)
            if text_on_first_generation:
                response = response + text_on_first_generation
            state["article"] = response
            return state
        except Exception as e:
            logger.exception(f"Error generating article: {str(e)}")
            raise Exception(f"Error generating article: {str(e)}") from None

    def revise(self, state: AgentState) -> AgentState:
        article = state["article"]
        critique_text = state["critique_text"]
        """Revise the article based on the provided critique."""
        logger.info("Revising article based on critique")
        prompt = [
            {
                "role": "system",
                "content": """
# Role
You are a skilled journalist tasked with revising an article based on the provided critique.
# Task
Your task is to carefully consider the feedback and make necessary improvements to enhance the article's quality.
""",
            },
            {
                "role": "user",
                "content": f"""
Original article:
{article}
Critique:
{critique_text}
Please revise the article based on the critique and reply only with the contents of the revised article without any notes.
""",
            },
        ]
        try:
            response = self.llm.invoke(prompt).content
            logger.info("Article revised successfully")
            state["article"] = response
            return state
        except Exception as e:
            logger.exception(f"Error revising article: {str(e)}")
            raise Exception(f"Error revising article: {str(e)}") from None

    def run(self, state: AgentState) -> AgentState:
        """Run the writer agent to write or revise the article."""
        if "critique_text" in state and state["critique_text"] is not None:
            state = self.revise(state)
            logger.info("Article revision completed")
        else:
            state = self.writer(state)
            logger.info("Article generation completed")
        return state


class CritiqueAgent:
    """Agent responsible for critiquing articles."""

    def __init__(self):
        self.llm_json = ChatOllama(model="llama3.1", temperature=0, format="json")

    def format_critique(self, critique: Union[Critique, dict]) -> str:
        """Format the critique as a string."""
        if isinstance(critique, dict):
            critique = Critique(**critique)
        formatted_lines = []
        for field_name, field_value in critique.dict().items():
            if field_value is not None:
                formatted_name = field_name.replace("_", " ").capitalize()
                formatted_lines.append(f"- **{formatted_name}**: {field_value}")
        return "\n".join(formatted_lines)

    def critique(self, state: AgentState) -> AgentState:
        """Provide a critique of the article."""
        article = state["article"]
        logger.info("Critiquing article")
        prompt = [
            {
                "role": "system",
                "content": f"""
# Role
You are an experienced editor with a keen eye for evaluating article quality.

## Task
Your task is to provide constructive feedback on the article in the specified JSON format, assessing its accuracy, grammar, relevance and overall quality.

After scoring each category on a scale of 1 to 10, you will calculate an overall score that reflects the general quality of the article.

## Example
If accuracy is 8, grammar is 7, and relevance is 9, the overall score should be around 8.

{parser.get_format_instructions()}
""",
            },
            {"role": "user", "content": article},
        ]
        try:
            critique_json = self.llm_json.with_structured_output(Critique).invoke(
                prompt
            )
            critique_text = self.format_critique(critique_json)
            logger.info(
                "Article critique completed. Critique Text: %s, Critique JSON: %s",
                critique_text,
                critique_json,
            )
            state["critique"] = critique_json
            state["critique_text"] = critique_text
            return state
        except Exception as e:
            logger.exception(f"Error critiquing article: {str(e)}")
            raise Exception(f"Error critiquing article: {str(e)}") from None

    def run(self, state: AgentState) -> AgentState:
        """Run the critique agent to critique the article."""
        state = self.critique(state)
        logger.info("Critique added to the article")
        return state


def check_output(state: AgentState) -> str:
    """Check the output of the critique and determine the next action."""
    if state["critique"] is not None:
        try:
            critique = state["critique"]
            if critique.overall_score > state["overall_score_required"]:
                logger.info("Article accepted")
                return "publish"
        except Exception as e:
            logger.exception(f"Error checking critique: {str(e)}")
    logger.info("Article needs revision")
    return "write"


class PublisherAgent:
    """Agent responsible for publishing articles."""

    def run(self, state: AgentState) -> AgentState:
        """Publish the final article."""
        logger.info(f"Final article:\n{state['article']}")
        return state


writer_agent = WriterAgent()
critique_agent = CritiqueAgent()
publisher_agent = PublisherAgent()

workflow = Graph()

workflow.add_node("write", writer_agent.run)
workflow.add_node("critique", critique_agent.run)
workflow.add_node("publish", publisher_agent.run)

workflow.add_edge("write", "critique")
workflow.add_conditional_edges(
    "critique",
    check_output,
    ["publish", "write"],
)

workflow.set_entry_point("write")
workflow.set_finish_point("publish")

graph = workflow.compile()

if __name__ == "__main__":
    topic = "Describing the subjective experience of skydiving from the perspective of someone who has done it multiple times."
    logger.info(
        f"Starting the article writing and critiquing workflow for topic: {topic}"
    )
    initial_state: AgentState = {
        "topic": topic,
        "critique_text": None,
        "critique": None,
        "article": None,
        "overall_score_required": 8,
    }

    config = {"configurable": {"thread_id": str(uuid4())}}
    result = None
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        result = event
        print(event)

    logger.info("Workflow completed successfully")
    print(f"Final result: {result}")
