---
title: floorcall
emoji: 📞
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
models:
- {model_id}
short_description: Voice-agent turn-taking decisions, naive vs floorcall
---

# floorcall

A millisecond decision layer for voice agents. It answers four questions from the conversation's
text, together:
- is the user done talking?
- was that "uh-huh" or a real interruption?
- what do they want, and does a bank's support agent handle it?
- do they need a human?

- **Replay:** eight scripted banking calls, each through a naive agent (answers after 800 ms of
  silence, stops for any speech) and through floorcall. This is the committed replay run, drawn
  as recorded. The calls are illustrative demos, not an evaluation set.
- **Results:** Tables A–D, the robustness table, the operating points and the tradeoff curves,
  rendered from the repository's result files.
- **Try it:** how to run live decisions locally. A Space with live inference needs a paid plan,
  so this one is static.

**Links:**
- Model: [{model_id}](https://huggingface.co/{model_id}), CC BY-NC-SA 4.0, non-commercial.
- Code and every number: [github.com/yashraz23/floorcall](https://github.com/yashraz23/floorcall),
  at commit `{code_sha}`.

The Space holds no real customer text: the replay scripts are written by hand.
