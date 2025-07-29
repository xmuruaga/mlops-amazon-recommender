from pydantic import BaseModel
from typing import List

from typing import Optional

class Recommendation(BaseModel):
    asin: str
    product_url: str
    thumbnail_url: str
    title: str = ""
    score: float
    avg_rating: Optional[float] = None

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
