from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    base64: str | None = None

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.path) == bool(self.base64):
            raise ValueError("Each image requires exactly one of path or base64")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=200_000)
    images: list[ImageInput] = Field(default_factory=list, max_length=8)
    model: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,127}$")
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh", "max", "ultra"] | None = None
    conversation_id: UUID | None = None
    workspace_dir: str | None = Field(default=None, min_length=1, max_length=4096)
    new_conversation: bool | None = None

    @model_validator(mode="after")
    def conversation_mode(self):
        if self.new_conversation is True and self.conversation_id is not None:
            raise ValueError("new_conversation=true cannot include conversation_id")
        if self.new_conversation is False and self.conversation_id is None:
            raise ValueError("new_conversation=false requires conversation_id")
        return self


class ChatResponse(BaseModel):
    conversation_id: str
    thread_id: str
    model: str | None
    reasoning_effort: str | None
    answer: str
    usage: dict = Field(default_factory=dict)
    elapsed_seconds: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[ChatRequest] = Field(min_length=1, max_length=32)


class JobBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requests: list[ChatRequest] = Field(min_length=1, max_length=1000)


class JobQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: list[UUID] = Field(min_length=1, max_length=1000)
