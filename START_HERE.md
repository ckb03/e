# Start the isolated research agent

Nothing starts automatically. The lab is `/workspace/role_confusion_research_lab` and `/workspace/e` must remain untouched.

## 1. Create a detached tmux session with Codex ready

```bash
cd /workspace/role_confusion_research_lab
./tmux_codex.sh start
```

This starts only the Codex TUI. It does not submit a prompt or consume research tokens.

## 2. Attach and explicitly start the goal

```bash
./tmux_codex.sh attach
```

Inside Codex, type:

```text
/goal Follow RUN_GOAL.md exactly. Work autonomously until its definition of done is satisfied.
```

Detach without stopping it with `Ctrl-b`, then `d`.

## 3. Check later

```bash
./tmux_codex.sh status
./tmux_codex.sh capture
```

The goal file instructs Codex to preserve state and wait if the rolling five-hour usage window is exhausted. Goal mode persists across automatic continuations, and tmux keeps the terminal alive after SSH logout. If the account service pauses rather than automatically resumes after reset, attach and type `/goal resume`; local setup cannot override account-side rate-limit behavior.

## Stop without deleting research files

```bash
./tmux_codex.sh stop
```

## Clean up everything later

After reviewing or copying results, remove the single lab directory yourself:

```text
/workspace/role_confusion_research_lab
```
