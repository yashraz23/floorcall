---
title: floorcall
emoji: 📞
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.11"
app_file: app.py
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
- **Try it:** one live decision on this Space's CPU, with the calibrated probabilities and the
  normalized state the model read.
- **Results:** Tables A–D, the robustness table and the operating points, rendered from the
  repository's result files.

**Links:**
- Model: [{model_id}](https://huggingface.co/{model_id}), CC BY-NC-SA 4.0, non-commercial.
- Code and every number: [github.com/yashraz23/floorcall](https://github.com/yashraz23/floorcall),
  at commit `{code_sha}`.

No API key is used, and the Space holds no real customer text: the replay scripts and examples
are written by hand.
