"""Real-task grading using the repository's existing math and IFEval implementations."""

import contextlib
import re
from decimal import Decimal, InvalidOperation

from langdetect import DetectorFactory

from open_instruct import if_functions, math_utils
from open_instruct.IFEvalG import instructions_registry


def ifeval(row, text):
    """Prompt/instruction strict and loose scores; errors propagate, never count as passes."""
    constraints = row["constraints"]
    if "func_name" in constraints:
        DetectorFactory.seed = 0
        arguments = {k: v for k, v in constraints.items() if k != "func_name" and v is not None}
        correct = bool(text.strip() and if_functions.IF_FUNCTIONS_MAP[constraints["func_name"]](text, **arguments))
        return dict(
            correct=correct,
            instruction_strict=[correct],
            instruction_loose=[correct],
            prompt_loose=correct,
            verifier="tulu3_legacy_training_constraint",
        )
    ids, kwargs = constraints["instruction_id"], constraints["kwargs"]
    lines = text.splitlines()
    variants = [text, text.replace("*", "")]
    for variant in ("\n".join(lines[1:]), "\n".join(lines[:-1]), "\n".join(lines[1:-1])):
        variants += [variant.strip(), variant.replace("*", "").strip()]
    strict, loose = [], []
    for instruction_id, arguments in zip(ids, kwargs, strict=True):
        checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        checker.build_description(**{k: v for k, v in (arguments or {}).items() if v is not None})
        if "prompt" in (checker.get_instruction_args() or []):
            prompt = row["prompt"]
            if isinstance(prompt, list):
                prompt = prompt[-1]["content"]
            checker.build_description(prompt=prompt)
        strict.append(bool(text.strip() and checker.check_following(text)))
        loose.append(any(variant.strip() and checker.check_following(variant) for variant in variants))
    return dict(correct=all(strict), instruction_strict=strict, instruction_loose=loose, prompt_loose=all(loose))


def grade(row, text):
    task = row["task"]
    if task == "ifeval":
        return ifeval(row, text)
    if task == "gsm8k":
        clean = re.sub(r"(?<=\d),(?=\d)", "", text)
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", clean)
        try:
            correct = bool(numbers) and Decimal(numbers[-1]) == Decimal(row["answer"].replace(",", ""))
        except InvalidOperation:
            correct = False
        return dict(correct=correct)
    if task != "math":
        raise ValueError(f"Unknown task {task}")
    # Same extraction families as ground_truth_utils.MathVerifier, without importing GPU/judge dependencies.
    answers = []
    boxed = math_utils.last_boxed_only_string(text)
    if boxed:
        with contextlib.suppress(AssertionError):
            answers.append(math_utils.remove_boxed(boxed))
    minerva = math_utils.normalize_final_answer(math_utils.get_unnormalized_answer(text))
    if minerva != "[invalidanswer]":
        answers.append(minerva)
    if not answers:
        latex = re.findall(r"\$([^$]+)\$", text)
        if latex:
            answers.append(math_utils.normalize_final_answer(latex[-1]))
    if not answers:
        answers = [math_utils.normalize_final_answer(text), text]
    diagnostics = {"symbolic_timeouts": 0, "symbolic_length_skips": 0}
    for answer in answers:
        if math_utils.hendrycks_is_equiv(answer, row["answer"]):
            return dict(correct=True, **diagnostics)
        # The existing symbolic comparator has a 5s timeout; cap malformed free-text inputs too.
        if len(answer) > 512:
            diagnostics["symbolic_length_skips"] += 1
            continue
        if math_utils.is_equiv(answer, row["answer"], diagnostics=diagnostics):
            return dict(correct=True, **diagnostics)
    return dict(correct=False, **diagnostics)
