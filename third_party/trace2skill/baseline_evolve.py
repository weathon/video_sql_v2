"""Responses structured-parse and flat-subskill bridge for pinned Trace2Skill."""

import dataclasses
import json
import logging
import re
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Literal

import httpx2
import weave
import wandb
import yaml
from openai import DefaultHttpxClient, OpenAI
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

import skill_evolver.parallel_evolving_agent as parallel
from skill_evolver.parallel_success_evolving_agent import CombinedParallelSkillEvolver
from skill_evolver.skill_evolving_agent import SkillEvolver
from tracing_bridge import RunMetrics, init_weave


MODEL = "gpt-5.6-luna"
HEADER = "---\nname: video-memory-qa\ndescription: Retrieve and verify evidence to answer questions about long videos.\n---\n\n"
BRIDGE_RULES = """
Environment adaptation: SKILL.md is the automatically loaded ROOT-SKILL guide.
references/*.md are on-demand subskills exposed through read_skill(name). Keep their
YAML frontmatter (name and description); name must equal the file stem. Never inline
all subskills into the root. Preserve the loading guide, catalog and read_skill API.
Use unquoted, one-line `name: ...` and `description: ...` fields for the harness catalog.
Change only SKILL.md or references/<name>.md. Never delete SKILL.md. Do not mention
gold answers or individual training examples in learned reusable guidance.
Use the supplied structured JSON schema, with empty strings for unused edit fields.
"""


class Edit(BaseModel):
    file: str
    op: Literal["insert_after", "insert_before", "append_to_section", "replace_in_section", "add_section", "delete_section", "create", "delete_file"]
    target_section: str
    target_text: str
    content: str
    old_text: str
    after_section: str


class PatchResponse(BaseModel):
    reasoning: str
    edits: list[Edit]
    changelog_entries: list[str]


class ContextExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)


class BridgeSkillEvolver(SkillEvolver):
    def validate_skill(self):
        valid, message = super().validate_skill()
        if not valid:
            return valid, message
        state = self.read_skill_state()
        if len(state["SKILL.md"].splitlines()) > self.max_skill_lines:
            return False, f"SKILL.md exceeds {self.max_skill_lines} lines"
        references = [name for name in state if name.startswith("references/")]
        if len(references) > self.max_references:
            return False, f"Too many subskills: {len(references)} > {self.max_references}"
        for name in references:
            match = re.match(r"^---\n(.*?)\n---(?:\n|$)", state[name], re.DOTALL)
            if match is None:
                return False, f"{name}: missing YAML name/description frontmatter required by read_skill"
            try:
                metadata = yaml.safe_load(match.group(1))
            except yaml.YAMLError as error:
                return False, f"{name}: {error}"
            if not isinstance(metadata, dict) or metadata.get("name") != Path(name).stem or not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
                return False, f"{name}: name must equal filename stem and description must be a nonempty string"
            lines = match.group(1).splitlines()
            if f"name: {metadata['name']}" not in lines or f"description: {metadata['description']}" not in lines:
                return False, f"{name}: catalog requires unquoted one-line name: and description: fields"
        for reference in re.findall(r"references/[\w.-]+\.md", state["SKILL.md"]):
            if reference not in state:
                return False, f"Dangling skill reference: {reference}"
        return True, "Upstream skill and harness catalog validation passed"

    def apply_edits(self, edits):
        for edit in edits:
            if edit.relative_path != "SKILL.md" and re.fullmatch(r"references/[a-z0-9]+(?:-[a-z0-9]+)*\.md", edit.relative_path) is None:
                raise ValueError(f"Edit outside the skill layout: {edit.relative_path}")
            if edit.relative_path == "SKILL.md" and edit.action == "delete":
                raise ValueError("The root loading guide cannot be deleted")
        super().apply_edits(edits)


class BaselineEvolver(CombinedParallelSkillEvolver):
    def __init__(self, run, client, metrics, **kwargs):
        super().__init__(client=client, **kwargs)
        self.metrics = metrics
        self._evolver = BridgeSkillEvolver(client=client, skill_dir=self.skill_dir, max_references=6)
        self.calls = run / "evolution/llm"
        self.calls.mkdir(parents=True, exist_ok=True)
        self.cache_lock = threading.Lock()
        self.cache = {}
        for path in self.calls.glob("*/parsed.json"):
            request = json.loads((path.parent / "request.json").read_text())
            self.cache[(request["instructions"], request["user_message"])] = path

    @weave.op(name="trace2skill.model", postprocess_inputs=lambda x: {k: v for k, v in x.items() if k != "self"})
    def _call_llm(self, system_prompt, user_message, max_continuations=2, tag="editting", expect_semantic=False):
        if expect_semantic or tag not in ("patch", "merged_patch"):
            raise ValueError(f"Unexpected baseline pipeline/schema: {expect_semantic}, {tag}")
        instructions = system_prompt + BRIDGE_RULES
        key = (instructions, user_message)
        conversation = [("system", instructions), ("user", user_message)]
        with self.cache_lock:
            cached = self.cache.get(key)
        if cached is not None:
            parsed = PatchResponse.model_validate_json(cached.read_text())
            response_text = "```json\n" + parsed.model_dump_json() + "\n```"
            return response_text, conversation + [("assistant", response_text)]
        directory = self.calls / uuid.uuid4().hex
        directory.mkdir()
        (directory / "request.json").write_text(json.dumps({"instructions": instructions, "user_message": user_message, "model": MODEL, "reasoning": {"effort": "high", "summary": "auto"}, "tag": tag}, ensure_ascii=False, indent=2) + "\n")
        messages = [{"role": "user", "content": user_message}]
        continuations = format_fixes = 0
        while True:
            number = continuations + format_fixes
            (directory / f"request-{number}.json").write_text(json.dumps(messages, ensure_ascii=False) + "\n")
            raw = self.client.responses.with_raw_response.parse(
                model=MODEL, instructions=instructions, input=messages,
                reasoning={"effort": "high", "summary": "auto"}, store=False,
                include=["reasoning.encrypted_content"], text_format=PatchResponse,
            )
            body = raw.content
            (directory / f"response-{number}.json").write_bytes(body)
            payload = json.loads(body)
            self.metrics.record_response(payload)
            # Preserve every output item, including encrypted reasoning, summaries
            # and assistant phase, through continuation and format repair.
            messages.extend(payload["output"])
            try:
                if payload["status"] == "incomplete" and payload["incomplete_details"]["reason"] == "max_output_tokens":
                    if continuations >= max_continuations:
                        raise RuntimeError(f"Trace2Skill continuation exhausted: {directory}")
                    continuations += 1
                    feedback = "Continue the incomplete response. Return the entire complete patch using the supplied schema."
                    messages.append({"role": "user", "content": feedback})
                    conversation.append(("user", feedback))
                    continue
                response = raw.parse()
                if response.status != "completed" or response.output_parsed is None:
                    raise RuntimeError(f"Trace2Skill response did not complete: {directory}")
                parsed = response.output_parsed
                for edit in parsed.edits:
                    if edit.file != "SKILL.md" and re.fullmatch(r"references/[a-z0-9]+(?:-[a-z0-9]+)*\.md", edit.file) is None:
                        raise ValueError(f"Invalid patch path: {edit.file}")
                    if edit.file == "SKILL.md" and edit.op == "delete_file":
                        raise ValueError("Cannot delete the root loading guide")
                response_text = "```json\n" + parsed.model_dump_json() + "\n```"
                valid, feedback = self._response_parse_feedback(response_text, tag)
                if not valid:
                    raise ValueError(feedback)
                patch = self._patch_from_data(parsed.model_dump())
                state = self.read_skill_state()
                created = {edit.file for edit in patch.edits if edit.op == "create"}
                for edit in patch.edits:
                    if edit.op != "create" and edit.file not in state and edit.file not in created:
                        raise ValueError(f"Patch targets a missing skill file: {edit.file}")
                if system_prompt != self._translation_system_prompt:
                    self._enforce_create_pairing([patch], state)
                if system_prompt in (self._translation_system_prompt, self._verification_system_prompt):
                    if system_prompt == self._translation_system_prompt and len(patch.edits) != 1:
                        raise ValueError("Translation must return exactly one edit")
                    self.run_apply_phase_programmatic(state, patch)
                break
            except (ValidationError, ValueError) as error:
                (directory / f"error-{number}.txt").write_text(str(error))
                print(f"Trace2Skill {tag} repair {format_fixes}: {error}; artifacts={directory}", flush=True)
                if format_fixes >= 2:
                    raise
                format_fixes += 1
                feedback = self._build_json_format_fix_prompt(tag, str(error))
                messages.append({"role": "user", "content": feedback})
                conversation.append(("user", feedback))
        (directory / "parsed.json").write_text(parsed.model_dump_json(indent=2) + "\n")
        with self.cache_lock:
            self.cache[key] = directory / "parsed.json"
        return response_text, conversation + [("assistant", response_text)]

    @staticmethod
    def _enforce_create_pairing(patches, skill_state):
        paired = CombinedParallelSkillEvolver._enforce_create_pairing(patches, skill_state)
        if sum(len(p.edits) for p in paired) != sum(len(p.edits) for p in patches):
            raise ValueError("Patch contains an unpaired create or dangling reference; repair the complete patch")
        return paired

    @staticmethod
    def _sanitize_translated_edits(skill_state, edits):
        result = CombinedParallelSkillEvolver._sanitize_translated_edits(skill_state, edits)
        if len(result) != len(edits):
            raise ValueError("Translated patch contains invalid/dangling edits; refusing to drop edits")
        return result

    run_map_phase = weave.op(name="trace2skill.map", postprocess_inputs=lambda x: {"records": len(x["records"])})(CombinedParallelSkillEvolver.run_map_phase)
    run_reduce_phase = weave.op(name="trace2skill.merge", postprocess_inputs=lambda x: {"patches": len(x["patches"])})(CombinedParallelSkillEvolver.run_reduce_phase)
    run_translation_phase = weave.op(name="trace2skill.translation", postprocess_inputs=lambda x: {"edits": len(x["final_patch"].edits)})(CombinedParallelSkillEvolver.run_translation_phase)
    run_apply_phase_programmatic = weave.op(name="trace2skill.apply", postprocess_inputs=lambda x: {"edits": len(x["final_patch"].edits)})(CombinedParallelSkillEvolver.run_apply_phase_programmatic)
    run_verification_phase = weave.op(name="trace2skill.verification", postprocess_inputs=lambda x: {})(CombinedParallelSkillEvolver.run_verification_phase)


@weave.op(name="trace2skill.evolution")
def evolve(run_directory):
    run = Path(run_directory)
    wandb_run = wandb.init(entity="3dsmile", project="video-sql-v2", id=run.name,
                           name=run.name, resume="must", dir=str(run))
    metrics = RunMetrics(wandb_run, "evolution", run)
    working = run / "evolution" / f"skill-{uuid.uuid4().hex}"
    (working / "references").mkdir(parents=True)
    initial = run / "skills/initial"
    references = sorted(p for p in initial.glob("*.md") if p.name != "ROOT-SKILL.md")
    root = HEADER + (initial / "ROOT-SKILL.md").read_text() + "\n## Subskill files\n\n"
    for source in references:
        shutil.copyfile(source, working / "references" / source.name)
        root += f"- [{source.stem}](references/{source.name})\n"
    (working / "SKILL.md").write_text(root)
    client = OpenAI(
        api_key=dotenv_values(Path(__file__).resolve().parents[2] / ".env")["PERSONAL_OPENAI_KEY"],
        base_url="https://api.openai.com/v1", max_retries=3,
        http_client=DefaultHttpxClient(
            timeout=httpx2.Timeout(connect=60, read=100000, write=100000, pool=100000),
            limits=httpx2.Limits(max_connections=1000, max_keepalive_connections=256),
        ),
    )
    try:
        evolver = BaselineEvolver(run, client, metrics, skill_dir=working, output_dir=run / "evolution",
                                 parse_failure_dir=run / "evolution/parse_failures", max_workers=128,
                                 max_references=6, skip_translation=False, patch_pipeline="json")
        records = json.loads((run / "combined-records.json").read_text())
        summary = evolver.run(records)
        valid, message = evolver._evolver.validate_skill()
        if not valid or summary["final_patch"] is None:
            raise RuntimeError(f"Evolution did not produce a valid skill: {message}")
        state = evolver.read_skill_state()
        for name, content in state.items():
            target = run / "distilled-skill" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        export = working / "harness"
        export.mkdir()
        root = re.sub(r"^---\n.*?\n---\n\n?", "", state["SKILL.md"], count=1, flags=re.DOTALL)
        (export / "ROOT-SKILL.md").write_text(root.replace("references/", ""))
        for name, content in state.items():
            if name.startswith("references/"):
                (export / Path(name).name).write_text(content)
        (run / "evolution-summary.json").write_text(json.dumps(summary, default=lambda x: dataclasses.asdict(x), ensure_ascii=False, indent=2) + "\n")
        export.rename(run / "skills/evolved")
        print(f"EVOLUTION COMPLETE {message}", flush=True)
    finally:
        client.close()
        wandb_run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parallel.ThreadPoolExecutor = ContextExecutor
    init_weave()
    with weave.attributes({"run": Path(sys.argv[1]).name, "skill_version": "initial-to-evolved"}):
        evolve(sys.argv[1])
    weave.finish()
