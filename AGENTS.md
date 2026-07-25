# llmoe - Agents Guide

## Project Overview

OLMoE inference engine written in C.

## Development Guidelines

### DO / DON'T
- **DO** write the project following C99 standard
- **DO NOT** read the entire project upfront to understand the codebase. Read files on demand as needed. Read only specific files relevant to the task at hand. This avoids wasting tokens on irrelevant code.
- **Linux only.** The project aims to run only on Linux OS, that said DO NOT spend effort trying handle cases for both platforms.
- **x86_64 only.** This project will only on a x86_64 machine. When I need you make some optimization like using "immintrin.h", DO NOT try to add support to ARM or RISC-V.
- **DO NOT** write/edit files when answering questions — only when asked to perform actions.
- **ALWAYS** run the full test suite (`make test`) after making changes — never run only the tests in scope.
- **ALWAYS** handle the warnings raised by the GCC compiler. If the best would be suppress it because it happens on a third-party library, ask me first.
- **ALWAYS** extract duplicated logic you encounter within the files you are modifying. Do not refactor duplication in untouched files.

# Code style

- Functions: 4-20 lines. Split if longer.
- Files: under 500 lines. Split by responsibility.
- One thing per function, one responsibility per module (SRP).
- Names: specific and unique. Avoid `data`, `handler`, `Manager`.
  Prefer names that return <5 grep hits in the codebase.
  Global Constants in UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`, `LAYER_SIZE`).
  functions, variable and local constants names in lower_snake_case (e.g., `load_weights`).
- Avoid introducing code duplication in new changes. When refactoring code within your current scope, extract shared logic into a function/module.
- Early returns over nested ifs. Max 3 levels of indentation.
- Function and variables must be written in English

## Comments

- Keep your own comments. Don't strip them on refactor — they carry
  intent and provenance.
- Write WHY, not WHAT. Skip `// increment counter` above `i++`.
- Docstrings on public functions: intent + one usage example.
- Reference issue numbers / commit SHAs when a line exists because
  of a specific bug or upstream constraint.
- When writing comments always write it in English

## Commits

- Use Conventional Commit message style when you've been requested to commit

## Documentation and Progress storage

- Create files under **docs** root project folder when need to document ideas, track bugs found, rationales, keypoints about the module itself. Use specific and unique file names, the file names must carry meaning in way to be predictable the content in.
