---
title: "Vision LLMs: how to send images to OpenAI-compatible APIs"
date: 2026-08-18
tags: [vision, llm, multimodal, openai-compatible, base64, image]
canonical: https://github.com/promptcracka/flippy
---

# Vision LLMs: how to send images to OpenAI-compatible APIs

Sending an image to a vision-capable LLM is a three-line change to your
existing chat-completion call. The shape is standard, but the implementation
details trip people up. This post is a complete reference.

## The wire shape

A vision message looks like a normal user message with `content` as a list
instead of a string:

```json
{
  "model": "...",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url",
       "image_url": {"url": "data:image/jpeg;base64,<BASE64_DATA>"}}
    ]
  }]
}
```

The model returns a normal text completion. That's it.

## The three things that bite people

### 1. Base64 is the standard way to send small images

For images under ~5MB, base64-in-a-data-URL is the most portable approach.
It works with every OpenAI-compatible provider. The catch: base64 inflates
the size by ~33%, and providers often cap image size around 20MB after
inflation.

```python
import base64
with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
url = f"data:image/jpeg;base64,{b64}"
```

### 2. The MIME type matters

The model uses the MIME type prefix to know the format. `image/jpeg` for
JPG, `image/png` for PNG, `image/webp` for WebP. Wrong MIME = the model
guesses (and often guesses wrong).

### 3. Some providers want URLs instead of base64

OpenAI's hosted API supports both:
- `{"type": "image_url", "image_url": {"url": "https://..."}}`
- `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`

Most OpenAI-compatible providers accept both. Some only accept one. When
in doubt, try base64 first — it's the more portable option for self-hosted
or smaller images.

## Resolution, cropping, and detail

Vision models resize large images down to a fixed input resolution (often
~1024px on the long edge). If your image has small but important details
(text in a screenshot, a small object in a wide photo), the model may not
see them. The fix is either:

- Crop to the relevant region before sending.
- Send a higher-resolution version if the provider supports it (some
  accept `image_url.detail: "high"`).

## When vision isn't the right tool

If you're classifying images at scale (every photo into one of 50
categories), a dedicated vision model (CLIP, SigLIP) is cheaper and more
accurate than asking a language model to do it. LLMs shine at *combined*
tasks: "look at this image and write a paragraph describing it," or "look
at this image and answer the user's question about it."

## A working example

The flippy hub has a `vision()` function at [`src/aihub.py`](https://github.com/promptcracka/flippy/blob/master/src/aihub.py):

```python
python src/aihub.py --vision photo.jpg "What is in this image?"
```

Verified working with `minimax-m3` on freeinference. The full pattern —
load image, base64 encode, build the messages list with the text + image
parts, POST to `/v1/chat/completions`, parse the response — is in the
function, ~15 lines of real code.

## Common provider quirks

- **freeinference**: works with base64 images out of the box.
- **Cloudflare Workers AI** (`@cf/meta/llama-3.2-11b-vision-instruct`): needs
  a one-time license acceptance in the Cloudflare dashboard before it
  works. The acceptance cannot be done via API.
- **NVIDIA NIM**: vision models exist (`meta/llama-3.2-11b-vision`) but are
  flaky — timeouts and 500s are common. Verify before relying on it for
  production.
- **Groq**: no vision models in the free tier.

The pattern that survives all of these: keep base64-encoding as your
fallback, and gate the actual provider choice behind a config flag you
can flip when one provider is having a bad day.
