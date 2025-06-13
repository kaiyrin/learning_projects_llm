import json, os
import ast
import re
from typing import List, Dict, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START
from config.llm_config import get_llm #ENTER YOUR LLM MODEL CONFIGURATIONS

# Load environment variables from .env file
_ = load_dotenv()



llm = get_llm()

class State(TypedDict):
    # User input
    book_name: str
    book_grade: str
    desc: Optional[int] = None
    
    # Book info
    book_info: Optional[str]
    chapter_breakdown: Optional[Dict[int, str]]

    # Current chapter
    chapter_number: int = 1
    chapter_name: Optional[str]
    chapter_text: Optional[str]

    # Accumulating results
    chapters: Dict[int, str] 
    qna: List[Tuple[str, str]]
    dialog: List[Tuple[str, str]] 
    max_chapter_no: int  = 1

def book_info_generator(state: State) -> dict:
    book_info_prompt = (
        f"""You are an expert educational content writer. 
        Write a concise (max 500 words) and engaging description for a textbook titled '{state["book_name"]}',
        appropriate for grade {state["book_grade"]} students. The description should be informative, age-appropriate, 
        and suitable as an introduction to the subject. No more than 100 words.
        Note.
        {state["book_grade"]} is he students grde level from 1 to 12 by British system."""
    )
    this_book_info = llm.invoke(book_info_prompt).content.strip()
    
    print({"book_info": this_book_info})
    
    
    return {"book_info": this_book_info}

def syllabus(state: State) -> dict:
    chapter_breakdown_prompt = (
        f"""You are an expert educational content writer.
        Given the following textbook information:
        - Book Name: {state["book_name"]}
        - Grade: {state["book_grade"]}
        - Book Description: {state["book_info"]}
        Create a sophisticated and comprehensive list of chapters (not more than 4) that would cover the full syllabus for this book.
        Each chapter should have a number and a clear, descriptive title.
        Return the result as a Python dictionary where the key is the chapter number (int) and the value is the chapter title (str).
        Example: {{1: "Introduction to ...", 2: "Fundamentals of ..."_}}, ...
        Execute only json output and without ```json\n
        """
    )
    chapters_output = llm.invoke(chapter_breakdown_prompt).content.strip()
 
    chapter_dict = ast.literal_eval(chapters_output)
   
        
    print("chapters_output:", chapter_dict, type(chapter_dict)) 
  
   
    return {"chapter_breakdown": chapter_dict, "max_chapter_no":  max(chapter_dict.keys(), default=0) if chapter_dict else 0}

def chaptertext_generator(state: State) -> dict:
    ch_no = state['chapter_number']  
    chapter_title = state.get('chapter_breakdown', {}).get(ch_no, {})  

    chapter_text_prompt = (
        f"""Write a textbook-style explanation for the chapter titled '{chapter_title}' 
        in the book '{state["book_name"]}' for grade {state["book_grade"]} students. 
        The explanation should be detailed, informative, and suitable for the target audience. 
        The explanation should be concise and not exceed 1000 words."""
    )
    chapter_text = llm.invoke(chapter_text_prompt).content.strip()
    state['chapters'][state['chapter_number']] = chapter_title 
    return {"chapter_name": chapter_title, "chapter_text": chapter_text, "chapter_number": ch_no}


def qna_generator(state: State) -> dict:
    ch_no = state["chapter_number"]
    chapter_title = state.get("chapter_name", f"Chapter {ch_no}")  # Use the current chapter name

    qanda_prompt = (
        f"Write up 2-3 questions and answers (Q&A) for chapter '{chapter_title}'and from the content of the '{state['chapter_text']} "
        f"of the {state['book_grade']} grade '{state['book_name']}' textbook. Format as 'Q: ... A: ...'."
    )
    raw_qna = llm.invoke(qanda_prompt).content
    qa_list = []
    current_q = None
    for line in raw_qna.split("\n"):
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:") and current_q is not None:
            a = line[2:].strip()
            qa_list.append((current_q, a))
            current_q = None  # Reset after pairing

    return {"qna": qa_list}

def dialog_generator(state: State) -> dict:
    ch_no = state["chapter_number"]
    chapter_title = state.get("chapter_name", f"Chapter {ch_no}")  # Use the current chapter name

    dialog_prompt = (
        f"Generate a short dialogue (2-3 exchanges) between a teacher (**Teacher**) and a student (**Student**) about the chapter '{chapter_title}' "
        f"from the {state['book_grade']} grade '{state['book_name']}' textbook. The dialogue should be engaging and educational."
    )
    raw_dialog = llm.invoke(dialog_prompt).content
    dialog_list = []
    current_speaker = None
    for line in raw_dialog.split("\n"):
        if line.startswith("**Teacher**:"):
            current_speaker = "Teacher"
            dialog_list.append((current_speaker, line[12:].strip()))
        elif line.startswith("**Student**:") and current_speaker == "Teacher":
            current_speaker = "Student"
            dialog_list.append((current_speaker, line[12:].strip()))
        elif current_speaker:
            dialog_list[-1] = (dialog_list[-1][0], dialog_list[-1][1] + " " + line.strip())
    return {"dialog": dialog_list}


def export_to_json(state: State) -> State:
    file_name = f"{state['book_name']}_{state['book_grade']}_content.json"
    ch_no = str(state["chapter_number"])

    # If file does not exist, write metadata and empty book_content
    if not os.path.exists(file_name):
        export_data = {
            "book_name": state["book_name"],
            "book_grade": state["book_grade"],
            "book_info": state["book_info"],
            "chapter_breakdown": state.get("chapter_breakdown", {}),
            "book_content": {}
        }
    else:
        with open(file_name, "r", encoding="utf-8") as f:
            export_data = json.load(f)

    # Update only the current chapter's content
    if "book_content" not in export_data:
        export_data["book_content"] = {}

    export_data["book_content"][ch_no] = {
        "chapter_title": state.get("chapter_name", ""),
        "chapter_text": state.get("chapter_text", ""),
        "qna": state.get("qna", []),
        "dialog": state.get("dialog", [])
    }

    # Write to file
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    print(f"Exported to {file_name}")


    print(f"Exported to {file_name}")
    return state

def increment_chapter(state: State) -> dict:
    state['chapter_number'] += 1
    return {"chapter_number": state['chapter_number']}

def chapter_loop_condition(state: State):
    if state['chapter_number'] >= state['max_chapter_no']:
        print("THIS IS END")
        return END
    else:
        return "increment_chapter"


# ----------- LANGGRAPH ------------
graph = StateGraph(State)

# NODE
graph.add_node("book_info_generator", book_info_generator)
graph.add_node("syllabus", syllabus)
graph.add_node("chaptertext_generator", chaptertext_generator)
graph.add_node("qna_generator", qna_generator)
graph.add_node("increment_chapter", increment_chapter)
graph.add_node("export_to_json", export_to_json)
graph.add_node("dialog_generator", dialog_generator)

# EDGE
graph.add_edge(START, "book_info_generator")
graph.add_edge("book_info_generator", "syllabus")
graph.add_edge("syllabus", "chaptertext_generator") 
graph.add_edge("chaptertext_generator", "qna_generator")
graph.add_edge("qna_generator", "dialog_generator")
graph.add_edge("dialog_generator", "export_to_json") 

# CONDITIONAL EDGE
graph.add_conditional_edges(
    "export_to_json",
    chapter_loop_condition,
    {
        END: END, 
        "increment_chapter": "increment_chapter"  
    }
)

graph.add_edge("increment_chapter", "chaptertext_generator")


compiled_graph = graph.compile()


try:
    img_data = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img_data)
    print("Graph saved to graph.png")
except Exception as e:
    print(f"Could not draw graph: {e}")


def run_generator(book_name: str, book_grade: str):

    initial_state: State = {
        "book_name": book_name,
        "book_grade": book_grade,
        "desc": None,
        "book_info": None,
        "chapter_breakdown": None,
        "chapter_number": 1, 
        "chapter_name": None,
        "chapter_text": None,
        "chapters": {},
        "qna": {},
        "max_chapter_no": 0 
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state

if __name__ == "__main__":

    final_result = run_generator(book_name="Literature", book_grade="8")
    print(json.dumps(final_result, indent=2))


