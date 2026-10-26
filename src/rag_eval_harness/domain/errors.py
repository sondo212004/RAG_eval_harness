"""AppError với `code` ổn định — client bắt theo `code`, `message` chỉ cho người đọc."""

from dataclasses import dataclass, field


@dataclass
class AppError(Exception):
    code: str
    message: str
    details: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
