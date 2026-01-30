from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    statement: str
    student_solution: str
    feedback: str
