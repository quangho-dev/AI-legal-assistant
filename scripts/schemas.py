from typing import Optional, List
from pydantic import BaseModel, Field

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
# 
class ChunkMetadata(BaseModel):
    
    law_type: Optional[str] = Field(default=None, description="Company name (lowercase, eg. 'amazon', 'apple', 'google',...)")
    publish_year: Optional[int] = Field(default=None, ge=1950, le=2050, description="Fiscal year of the document")

    model_config = {"use_enum_values": True}


class RankingKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Generate Exactly 5 legal keywords related to user query", min_length=5, max_length=5)

    