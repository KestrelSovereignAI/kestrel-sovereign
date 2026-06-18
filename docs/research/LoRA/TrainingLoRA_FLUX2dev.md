---
type: Research Note
title: '**Training LoRA on the FLUX 2 [dev] Diffusion Model**'
description: '**FluxGym (Next Diffusion)** – FluxGym is a user-friendly web UI specifically
  designed for training LoRA adapters on FLUX models [1](https://www.nextdiffusion.ai/tutorials/how-t...'
resource: /docs/research/LoRA/TrainingLoRA_FLUX2dev.md
tags:
- docs
- research
- research-note
timestamp: '2026-06-18T00:00:00Z'
status: snapshot
owner: documentation
canonical: false
generated: false
privacy: public
---

# **Training LoRA on the FLUX 2 [dev] Diffusion Model**

## **Tools and Templates for FLUX 2 LoRA Training**



**FluxGym (Next Diffusion)** – FluxGym is a user-friendly web UI specifically designed for training LoRA
adapters on FLUX models [1](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Creating%20consistent%20AI%20characters%20is,fast%20and%20accessible%20to%20anyone) . Next Diffusion provides a ready-to-use FluxGym **RunPod template**, so you
can deploy a cloud GPU pod with FluxGym pre-installed in one click [2](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=of%20the%20most%20reliable%20ways,fast%20and%20accessible%20to%20anyone) . This removes the need for manual
setup – the Docker image comes with all training scripts and the FluxGym interface baked in [3](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Image%3A%20Uploaded%20image) . Once
launched, FluxGym’s interface lets you upload images, caption them, and configure LoRA training
parameters through a browser UI. A similar **Vast.ai FluxGym template** is also available for quick setup on
Vast; you can rent a machine and launch FluxGym via a pre-configured Docker image there as well [4](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=an%20RTX%204090%20at%20%240,is%20sufficient%20for%20budget%20training) .



[1](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Creating%20consistent%20AI%20characters%20is,fast%20and%20accessible%20to%20anyone)



[2](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=of%20the%20most%20reliable%20ways,fast%20and%20accessible%20to%20anyone)



[3](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Image%3A%20Uploaded%20image)



[4](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=an%20RTX%204090%20at%20%240,is%20sufficient%20for%20budget%20training)



**Ostris AI-Toolkit** – An alternative is the AI-Toolkit by Ostris (RunComfy cloud or self-hosted), which provides
a robust interface (via Jupyter or web UI) for fine-tuning diffusion models including FLUX.2 LoRAs [5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit) . The
AI-Toolkit supports advanced memory optimizations and was explicitly recommended in the official FLUX.2
fine-tuning guide as a convenient option [5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit) . Using AI-Toolkit typically involves cloning the GitHub repo and
installing requirements, then launching its UI or scripts (as shown in community tutorials). For example,
Geronimo’s Medium guide on FLUX.1-dev walks through using AI-Toolkit on a RunPod Jupyter instance
(cloning the repo, uploading data, and running the training script) [6](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=We%20first%20download%20the%20Ostris%E2%80%99,use%20any%20other%20directory%20here) [7](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=Step%204%3A%20Login%20to%20Hugging,Face) . A similar workflow applies to
FLUX.2, with the main difference being pointing to the FLUX.2 base model and adjusting settings for the
larger model.



[5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit)



[5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit)



[6](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=We%20first%20download%20the%20Ostris%E2%80%99,use%20any%20other%20directory%20here) [7](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=Step%204%3A%20Login%20to%20Hugging,Face)



**SimpleTuner** – SimpleTuner (by bghira) is a general fine-tuning kit with a web dashboard that supports
many models, including FLUX.2 [8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) [9](https://github.com/bghira/SimpleTuner#:~:text=%2A%20TwinFlow%20Few,matching%20transformer%20with%20Chroma) . It provides a unified pipeline and heavy memory optimizations so
that _“most models [are] trainable on a 24G GPU, [and] many on 16G with optimizations”_ [10](https://github.com/bghira/SimpleTuner#:~:text=LyCORIS%20,weights%20for%20improved%20stability%20and) . SimpleTuner has
specific support for FLUX.2’s architecture (e.g. it recognizes the Mistral-3 text encoder and uses
quantization) [8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) . There are quickstart guides for FLUX.1 and **FLUX.2** included in its documentation [9](https://github.com/bghira/SimpleTuner#:~:text=%2A%20TwinFlow%20Few,matching%20transformer%20with%20Chroma),
meaning you can follow step-by-step instructions to fine-tune a FLUX.2 LoRA using this toolkit. This is a
more advanced option but favored by some community members for its efficiency and flexibility.



[8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) [9](https://github.com/bghira/SimpleTuner#:~:text=%2A%20TwinFlow%20Few,matching%20transformer%20with%20Chroma)



[10](https://github.com/bghira/SimpleTuner#:~:text=LyCORIS%20,weights%20for%20improved%20stability%20and)



[8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) [9](https://github.com/bghira/SimpleTuner#:~:text=%2A%20TwinFlow%20Few,matching%20transformer%20with%20Chroma)



**Hugging Face Diffusers Scripts** – As of late 2025, Hugging Face’s Diffusers library officially supports FLUX.2
and provides training scripts (e.g. `train_dreambooth_lora_flux2.py` ) with built-in optimizations. A

recent Hugging Face blog post on FLUX-2 details how to fine-tune LoRAs on FLUX.2 using Diffusers,
leveraging features like remote text encoder processes, gradient checkpointing, and 4-bit/8-bit quantization
to fit the model in memory [11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing) . The blog even shares example training commands for DreamBoothstyle LoRA training on FLUX.2 (with flags for FP8 training, offloading, etc.) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22) [14](https://huggingface.co/blog/flux-2#:~:text=,ancient%20looking%20trollface%2C%20%27the%20shitposter) . This indicates that
**actively maintained** scripts and documentation exist for FLUX.2 LoRA training in the diffusers ecosystem,
which can be a starting point if you prefer Python scripts over a custom UI.



[11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing)



[13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22) [14](https://huggingface.co/blog/flux-2#:~:text=,ancient%20looking%20trollface%2C%20%27the%20shitposter)



1


## **Step-by-Step Guides and Examples**

There are several **step-by-step tutorials** and community examples focused on FLUX LoRA training:



**Next Diffusion’s FluxGym RunPod Guide (June 2025)** – A comprehensive tutorial shows how to
launch the Next Diffusion – FluxGym template on RunPod and train a character LoRA on **FLUX.2-dev**

[15](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character) . It covers everything from setting up a RunPod account to deploying the pod, initializing the

VSCode workspace, and using the FluxGym web UI. Notably, it recommends using a high-VRAM GPU
(like an RTX 4090) and even notes that at the time of writing, a 4090 on RunPod Community Cloud
costs about **$0.34/hour** [16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price) – a good balance of price and performance. The guide then walks
through dataset preparation (15–20 images of the target character, using 1024×1024 resolution and
even augmenting with Flux “Kontext” image-to-image for variety) [17](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=A%20well,a%20diverse%20range%20of%20shots) [18](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Boost%20Dataset%20Variety%20with%20Flux,Kontext%20Dev) . It demonstrates how to
auto-caption images in FluxGym (using Florence-2 captions and refining them) and suggests LoRA

training settings (e.g. ~10 repeats, ~12 epochs for ~15 images) [19](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=) . Finally, it shows testing the LoRA
in ComfyUI after training. This guide is very **beginner-friendly**, with screenshots of each step and
tips (for example, reminding you to keep the backend terminal running while the UI is open [20](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=match%20at%20L226%20,for%20debugging%20or%20monitoring%20purposes) ).







[15](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character)



[16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price)



[17](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=A%20well,a%20diverse%20range%20of%20shots) [18](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Boost%20Dataset%20Variety%20with%20Flux,Kontext%20Dev)



[19](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=)



[20](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=match%20at%20L226%20,for%20debugging%20or%20monitoring%20purposes)



**Vast.ai + FluxGym Guides** – Similar step-by-step guides exist for Vast.ai if you prefer that platform.
One blog post demonstrates training a _Flux.1-dev_ LoRA for under \$1 using Vast.ai and FluxGym [21](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Training%20a%20Flux,model%20without%20breaking%20the%20bank)

[22](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Key%20Benefits%20of%20Training%20Flux,Dev%20LoRA) . It details selecting a cheap GPU (as low as an RTX 3060 with 12GB for ~$0.07/hour) and using a

FluxGym Docker image on Vast [23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) . The guide shows how even a small dataset (10–15 images) can
be used to fine-tune a style LoRA in about 2–3 hours (which would cost well under \$1 on a 3060) [24](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=such%20as%20loss%2C%20to%20track,training) .
While this example was FLUX.1, the process on FLUX.2 is analogous – just expect that FLUX.2 may
take longer or benefit from a more powerful GPU due to its size. There’s also a **“Super Simple LoRA**
**Training on Vast”** community guide (GitHub repo _simple-flux-lora-training_ ) that provides a minimal
set of steps to launch a Vast instance and run training via SSH, aimed at flux-dev models. In
summary, if RunPod is not your preference, Vast.ai can be a cost-effective alternative, and the
community has documented the process to get Flux LoRA training running there as well [25](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [26](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=docker%20pull%20aidockorg%2Ffluxgym) .







[21](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Training%20a%20Flux,model%20without%20breaking%20the%20bank)



[22](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Key%20Benefits%20of%20Training%20Flux,Dev%20LoRA)



[23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training)



[24](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=such%20as%20loss%2C%20to%20track,training)



[25](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [26](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=docker%20pull%20aidockorg%2Ffluxgym)



**RunComfy Flux.2 LoRA Guide** – The RunComfy site (affiliated with Ostris’ AI-Toolkit) published a
detailed **FLUX.2 [dev] LoRA Training Guide** in 2025. This guide is more advanced, diving into ideal
settings for FLUX.2 LoRAs. It discusses how to configure the AI-Toolkit UI for FLUX.2 and provides
**recommendations by GPU VRAM tier** (16–24GB, 32–48GB, 64GB+) [27](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,4070%20Ti%2C%204080%2C%204090) [28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) . For example, on a 24GB
GPU they suggest turning on “Low VRAM” and layer offloading options, using FP8/4-bit quantization,
batch size = 1 with grad accumulation, and keeping resolution around 896–1024px to avoid OOM
issues [29](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [30](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20QUANTIZATION%20panel%2C%20set,float8%20%28default) . In contrast, a 48GB GPU can handle 1024px images more comfortably, possibly batch
size 2, and use rank 32 LoRAs with ~1000-3000 training steps for high-quality results [28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) [31](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20QUANTIZATION%2C%20set%20both%20Transformer,while%20comfortably%20fitting%20the%20model) . This
guide is extremely useful for understanding **performance tuning** – it highlights how to squeeze
FLUX.2 training onto smaller GPUs and what trade-offs to expect. It also covers dataset design
(recommending ~20–60 images for a character LoRA, if possible, though good results can be
achieved with a few dozen carefully chosen images) [32](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=From%20available%20FLUX%20examples%20and,similar%20LoRA%20trainings) [33](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=Official%20FLUX%20LoRA%20examples%20on,a%20few%20dozen%20well%E2%80%91chosen%20images) . If you plan to train a FLUX.2 LoRA from
scratch, reviewing these best practices can save a lot of trial and error.







[27](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,4070%20Ti%2C%204080%2C%204090) [28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs)



[29](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [30](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20QUANTIZATION%20panel%2C%20set,float8%20%28default)



[28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) [31](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20QUANTIZATION%2C%20set%20both%20Transformer,while%20comfortably%20fitting%20the%20model)



[32](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=From%20available%20FLUX%20examples%20and,similar%20LoRA%20trainings) [33](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=Official%20FLUX%20LoRA%20examples%20on,a%20few%20dozen%20well%E2%80%91chosen%20images)




- **Community Videos & Notebooks** – There are YouTube tutorials (e.g. _“How to train your FLUX.2 dev_

_consistent character LoRA”_ by AI enthusiasts) and even Colab notebooks shared in communities. For
instance, some users in Reddit/Facebook groups have shared Colab notebooks to train Flux LoRAs
on Colab Pro (A100 GPUs), lowering the entry barrier for those without their own GPU. These


2


typically automate steps like logging into Hugging Face, downloading the FLUX.2 base model, and
running an AI-Toolkit or Diffusers training script. While such community notebooks/templates are
available, it’s important to ensure they are up-to-date due to FLUX.2’s rapid development.
Nonetheless, they illustrate that **working examples** exist – you’re not the first attempting FLUX.2
LoRA training, and you can often find a template to start from rather than writing everything from

scratch.

## **FLUX.2 vs Other Models: Compatibility and Performance**


Training LoRAs on FLUX.2 [dev] has some key differences in requirements compared to Stable Diffusion or

even SDXL:


    - **Model Size and VRAM Needs:** FLUX.2 is a _much_ larger model – roughly 32B parameters – whereas



SDXL is about 3.5B. Running the full FLUX.2 (image generator + text encoder) can consume **60–80 GB**
**of VRAM** just for inference [34](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=FLUX.2%20,batch%20sizes%20and%20latent%2Ftext%20caching) [35](https://huggingface.co/blog/flux-2#:~:text=LoRA%20fine) . By contrast, SDXL or Stable Diffusion 1.5 can run on ~8–16GB for
inference. This means FLUX.2 LoRA training is **significantly more memory-hungry** . In practice,
training requires heavy use of memory-saving techniques: 8-bit/4-bit quantization, CPU offloading of
weights, gradient checkpointing, caching latents, etc. [5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing) . For example, the official diffusers
guide notes that even with CPU offload, FP8 training, and other tricks, they still needed ~20GB or
more GPU memory for a batch size of 2 at 1024px [11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22) . The bottom line is that **FLUX.2 pushes**
**the limits of GPU memory**, so expect slower training and more fiddling with settings if you’re on a
consumer GPU. It’s often recommended to use at least a 24GB card (RTX 3090/4090) or better.

Community members have reported that 16GB cards _can_ work with aggressive optimizations, but it’s
“tight” and mostly limited to very small LoRAs (few images, batch 1) with a risk of out-of-memory
errors [36](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [37](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=streamed%20from%20CPU%20RAM%20instead,the%20GPU%20the%20whole%20time) . On the upside, cloud providers like RunPod now offer high-memory GPUs – e.g. **80GB**
**A100 or H100** – which make FLUX.2 training much easier (these were used in some published guides
to comfortably run 1024×1024 training with batch size 2–4) [38](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,H100%2C%20H200%20on%20RunComfy) [39](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=) .



[34](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=FLUX.2%20,batch%20sizes%20and%20latent%2Ftext%20caching) [35](https://huggingface.co/blog/flux-2#:~:text=LoRA%20fine)



[5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing)



[11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22)



[36](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [37](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=streamed%20from%20CPU%20RAM%20instead,the%20GPU%20the%20whole%20time)



[38](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,H100%2C%20H200%20on%20RunComfy) [39](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=)



**Text Encoder and Architecture:** Unlike Stable Diffusion which uses CLIP encoders, FLUX.2 uses a
**Mistral 3.1** small language model as its text encoder [40](https://huggingface.co/blog/flux-2#:~:text=First%2C%20instead%20of%20two%20text,known%20to%20be%20more%20beneficial) [41](https://huggingface.co/blog/flux-2#:~:text=Inference%20With%20Diffusers) . Moreover, FLUX.2’s “double-stream +
single-stream” transformer architecture is unique (an evolution of Flux.1’s DiT model). This doesn’t
change how you train a LoRA fundamentally, but it means the training code or UI must specifically
support FLUX.2’s model class. Thankfully, tools like FluxGym, AI-Toolkit, and SimpleTuner have
already integrated this support. One architectural difference is the **autoencoder/VAE** : FLUX.2
introduced a new `AutoencoderKLFlux2` with **32 latent channels** (versus 16 in Stable Diffusion’s

VAE). This yields higher detail and fidelity for 1024px images, but at the cost of roughly doubling
latent-related memory usage [42](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=FLUX,is%20designed%20for%201024%C3%971024%2B%20work) . For LoRA training, this means handling higher resolutions is more
expensive in FLUX.2. In practice, guides recommend training at 768 or 1024 resolution for FLUX.2 to
leverage its detail, whereas SDXL LoRAs might often train at 512 or 640. The larger latent also slightly
changes how many images you can fit in memory at once (again reinforcing the need for small batch

sizes).







[40](https://huggingface.co/blog/flux-2#:~:text=First%2C%20instead%20of%20two%20text,known%20to%20be%20more%20beneficial) [41](https://huggingface.co/blog/flux-2#:~:text=Inference%20With%20Diffusers)



**Training Speed:** Because of the model’s scale, each training step is slower on FLUX.2. Community
members note that LoRA training which might take 1 hour on SD1.5 could take several hours on
Flux, even on strong GPUs. One user discussion about Flux LoRA noted needing to reduce learning
rates and increase training steps carefully to avoid _“burning”_ the output (over-saturation) around
500+ steps [43](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=Just%20starting%20this%20so%20people,for%20training%20flux%20loras) . LoRAs on FLUX may need more gradual training or regularization (like “Differential







[43](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=Just%20starting%20this%20so%20people,for%20training%20flux%20loras)



3


Output Preservation”, a technique mentioned in RunComfy’s guide to keep the base model’s
capabilities outside the trigger term) to maintain stability. **Compatibility-wise**, once trained, a Flux2
LoRA is only usable with the FLUX.2 model (just as SDXL LoRAs only work on SDXL). The format of the
LoRA weights can be similar (often stored in Diffusers format or as a `.safetensors` with the LoRA

layers), but some UIs like ComfyUI might require a slightly different key naming for Flux LoRAs.
SimpleTuner, for instance, can save LoRAs in either standard Diffusers layout or in “ComfyUI-style
keys” for Flux models [44](https://github.com/bghira/SimpleTuner#:~:text=%28guide%29%20,detect%20ComfyUI%20inputs) [45](https://github.com/bghira/SimpleTuner#:~:text=%2A%20HiDream%20MoE%20,CFG%20reintroduction%20for%20distilled%20models), indicating that the community has solved the compatibility issues for
using LoRAs in generation pipelines (ComfyUI, Diffusers, etc.).



**Performance vs SDXL:** Users who have tried both report that FLUX.2 can achieve very **crisp details**
**and better text rendering** than SDXL when fine-tuned properly, especially at high resolution.
However, the effort to train is greater. One Redditor compared FLUX 2 dev vs another model and
noted the improvement in certain qualities (e.g. **text clarity in images** is a known strength of Flux).
So while SDXL LoRAs might be easier/faster to train (and SDXL needs ~24GB VRAM for training at
most), FLUX.2 LoRAs have the potential payoff of a more powerful base model – but you’ll likely
spend more time and money on the training process due to the heavier architecture. It’s worth
noting that FLUX.2 was not designed as a drop-in SD replacement but as a new paradigm combining
image generation and editing capabilities [46](https://huggingface.co/blog/flux-2#:~:text=FLUX,tuning), so LoRAs can latch onto both text-to-image and
image-to-image aspects. In summary, expect a FLUX.2 LoRA pipeline to require **more planning**
(small batches, good captions, possibly more training steps at lower learning rates) compared to an
SDXL LoRA, but the community consensus is that the results can be rewarding if those hurdles are

overcome.







[46](https://huggingface.co/blog/flux-2#:~:text=FLUX,tuning)


## **FLUX.2 LoRA Training Pipeline Differences and Tips**

When setting up your LoRA training pipeline for FLUX.2, keep these differences and tips in mind:


    - **Base Model Access:** The FLUX.2 [dev] model is hosted on Hugging Face ( _`black-forest-labs/`_


_`FLUX.2-dev`_ ) under a gated license. You **must log in and agree to the terms** on HF before you can


download or use it in training [47](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main#:~:text=By%20clicking%20,Up%20to%20review%20the) . In practice, this means if you’re using any scripts or tools outside
of RunPod’s template, you’ll need to authenticate with your Hugging Face token. Many training tools
allow setting an `HF_TOKEN` (for example, Ostris’ AI-Toolkit expects you to put your token in a `.env`

file for it to use [48](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=1) [49](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=%E2%80%A2%20%201y%20ago) ). If you see errors like _“Access to model is restricted or not a valid model id”_ [50](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=Access%20to%20model%20black,be%20authenticated%20to%20access%20it),
it usually means the Hugging Face login/token wasn’t configured correctly. Once you’ve agreed to
the FLUX license and logged in, the training script will fetch the model weights (which are large,
~12-16 GB for FP16 or ~8 GB for 8-bit quantized weights). Note that RunPod’s FluxGym template or
RunComfy’s cloud toolkit often have the model preloaded or cached once you log in, saving you from
downloading every run.



**Data Preparation and Format:** The workflow for data is quite similar to Stable Diffusion LoRAs: you
gather a dataset of images and (ideally) caption them with a unique trigger word. In FluxGym and AIToolkit, you’ll upload images and provide a caption (either manually or via an AI captioner like
Florence or BLIP). The **captions should include your chosen trigger word** for the concept/style
you’re teaching [51](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Since%20we%20set%20ch3rrybl0nde%20as,so%20it%E2%80%99s%20worth%20reviewing%20them) . A common best practice (as noted in the RunComfy guide) is to prepend the
trigger word to every caption and then describe the image without redundantly stating what the
trigger represents [52](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=captions%20might%20be%20a%20bit,so%20it%E2%80%99s%20worth%20reviewing%20them) . For example, if your LoRA’s trigger is `mystyle`, a caption could be:


“ **mystyle**, a landscape painting of mountains at sunset, vivid colors, oil painting”. The trigger


4







[51](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Since%20we%20set%20ch3rrybl0nde%20as,so%20it%E2%80%99s%20worth%20reviewing%20them)



[52](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=captions%20might%20be%20a%20bit,so%20it%E2%80%99s%20worth%20reviewing%20them)


( `mystyle` ) is nonsense to the base model initially, and the LoRA will learn to associate it with the

style or subject. This process and the data folder format (images and `.txt` caption files) are

essentially the same as for SD or SDXL LoRA training. The differences come in _quantity_ and _quality_ :
FLUX.2 can capture fine details, so high-resolution **1024×1024** training images are beneficial
(whereas many SD LoRA guides used 512px images). Also, FLUX.2 can leverage more images if
available – it’s been shown to handle **hundreds of images** for LoRA fine-tunes in official examples

[53](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,source%2C%20target%2C%20instruction%29%20triples), though personal projects often use a few dozen due to time constraints. Keep your dataset

well-curated and use a variety of angles/lighting if training a character, as usual.



**Training Scripts and Settings:** If using **FluxGym UI**, much is handled for you via dropdowns and
sliders. You would select “Base Model: `flux-dev` ” in the UI (this corresponds to FLUX.2 dev) [54](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character),


give your LoRA a name and trigger, then adjust parameters. For FLUX.2, typical LoRA
hyperparameters aren’t very different: LoRA Rank 16 or 32 is common (higher ranks capture more
but use more VRAM), learning rates around **1e-4** (0.0001) are a solid default [55](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=this%20keeps%20quality%20high%20while,comfortably%20fitting%20the%20model) [56](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer), and total
training steps might range from 1000 up to 3000 depending on dataset size [28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) [57](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer) . One difference

is that you’ll likely use **batch size = 1** almost always on FLUX.2 unless you have a top-tier GPU. On SD
one might do batch 2 or 4; on FLUX.2, memory is scarce so you accumulate gradients instead of
increasing batch. Community guides suggest using gradient accumulation to simulate a larger batch
if needed (e.g. accumulate 2-4 steps on a 24GB GPU) [58](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=option%20if%20your%20build%20exposes,float8%20%28default) [59](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20TRAINING%20panel%2C%20keep,need%20a%20larger%20effective%20batch) . Also, enabling features like _caching text_
_embeddings_ (to avoid re-encoding captions every step) can save a bit of memory [60](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20DATASETS%20panel%2C%20prefer,VRAM%20once%20captions%20are%20encoded) . If using
Diffusers or SimpleTuner scripts, you’ll see flags to enable many of the optimizations listed above
( `--offload`, `--cache_latents`, `--gradient_checkpointing`, `--fp8` or


`--bitsandbytes 4bit`, etc.) [61](https://huggingface.co/blog/flux-2#:~:text=,bit) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing) . These are essentially mandatory for FLUX.2 on anything less


than a server-grade GPU.







[54](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character)



[55](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=this%20keeps%20quality%20high%20while,comfortably%20fitting%20the%20model) [56](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer)



[28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) [57](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer)



[58](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=option%20if%20your%20build%20exposes,float8%20%28default) [59](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20TRAINING%20panel%2C%20keep,need%20a%20larger%20effective%20batch)



[60](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20DATASETS%20panel%2C%20prefer,VRAM%20once%20captions%20are%20encoded)



`--bitsandbytes 4bit`, etc.) [61](https://huggingface.co/blog/flux-2#:~:text=,bit) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing) . These are essentially mandatory for FLUX.2 on anything less



**Recommended GPU Specs and Cloud Pricing:** In general, **24 GB VRAM** is recommended as a
baseline for FLUX.2 LoRA training [62](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=3,or%20a%20similarly%20powerful%20GPU) . An NVIDIA RTX 4090 (24GB) was explicitly recommended in
NextDiffusion’s guide for “serious training workloads” [62](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=3,or%20a%20similarly%20powerful%20GPU) . On RunPod, such a GPU currently costs on
the order of **$0.30–$0.45 per hour** on-demand (e.g. ~$0.34/hr as quoted in June 2025) [16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price) . If budget
is a concern, you _can_ use smaller GPUs (RTX 3080/3060 with 10–12 GB) for simple LoRAs – one Vast.ai
example used a 12GB RTX 3060 at **$0.07/hr** successfully [23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) – but expect to limit resolution and
training speed significantly. For larger projects, cloud providers also offer A100 40GB, A100 80GB, or
even multi-GPU setups. The RunComfy guide references **H100 80GB and “H200” 141GB pods** [63](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=You%20can%20run%20this%20FLUX,LoRA%20workflow%20in%20two%20ways) –
the latter presumably a multi-GPU or special config – which can be rented if you truly need to push
FLUX.2 to its limits. Of course, those come at a premium price (several dollars per hour). Vast.ai often
has cheaper rates than RunPod for equivalent hardware (community members noted a 4090 on Vast
might be ~$0.50/hr when RunPod was ~$0.70/hr for similar, though prices fluctuate [64](https://weirdwonderfulai.art/resources/lora-training-with-ai-toolkit-on-vast-ai-under-50-cents/#:~:text=cents%20weirdwonderfulai,than%20using%20it%20on%20RunPod) ). **In**
**summary**, a single 4090 is a good sweet spot for FLUX.2 LoRAs: it’s affordable per hour and with the
proper optimizations can handle 1024px training with LoRA rank 16–32. Expect maybe ~1–3 hours of
training for a small dataset (which might cost only a dollar or two), or more like 8–10 hours if you
train on ~50–100 images or more steps (still under ~$5 on a community GPU). Always monitor the
GPU memory usage and adjust settings to avoid out-of-memory errors – the first few minutes of
training will usually tell you if your settings are too ambitious for the GPU.







[62](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=3,or%20a%20similarly%20powerful%20GPU)



[62](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=3,or%20a%20similarly%20powerful%20GPU)



[16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price)



[23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training)



[63](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=You%20can%20run%20this%20FLUX,LoRA%20workflow%20in%20two%20ways)



[64](https://weirdwonderfulai.art/resources/lora-training-with-ai-toolkit-on-vast-ai-under-50-cents/#:~:text=cents%20weirdwonderfulai,than%20using%20it%20on%20RunPod)



**Community Experience:** The community has accumulated a lot of **trial-and-error wisdom** on
training FLUX LoRAs. For example, in one GitHub discussion, users found that using a _quantized_ base
model (4-bit) was essential to experiment with higher batch sizes or fancy optimizers on Flux [65](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=) .







[65](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=)



5


Others emphasized that **learning rate scheduling** and not over-training are important; FLUX can
“burn” (produce distorted outputs) if overfit, so checking sample outputs every few hundred steps (a
feature FluxGym supports) is wise. Also, because FLUX.2 is so powerful, even ~10 images can impart
a style – but you may need to use a low learning rate or fewer repeats to avoid the LoRA memorizing
too hard. Many users report good results with ~20–30 images at 1024px, ~1000 steps at 1e-4 LR for
styles; for distinct characters, sometimes more steps or images (50+) are used [32](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=From%20available%20FLUX%20examples%20and,similar%20LoRA%20trainings) [53](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,source%2C%20target%2C%20instruction%29%20triples) . If you run
into problems, resources like the r/StableDiffusion subreddit or the Discords for these tools often
have folks who’ve done FLUX.2 LoRAs. Common issues include the HuggingFace authentication
(solved by `.env` token as discussed), and CUDA OOM errors (solved by toggling those memory


options or reducing resolution). The good news is that **FLUX.2 LoRA training has been done**
**successfully by many**, so whether you choose a template like FluxGym or a script-based approach,
you can accelerate your progress by following these templates and tips instead of reinventing the

wheel.

## **References**



Next Diffusion, _“How to Train a Flux LoRA with FluxGym on RunPod”_ (June 25, 2025) [1](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Creating%20consistent%20AI%20characters%20is,fast%20and%20accessible%20to%20anyone) [16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price)
RunComfy (Ostris), _“FLUX.2 [dev] LoRA Training Guide”_ [29](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [31](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20QUANTIZATION%2C%20set%20both%20Transformer,while%20comfortably%20fitting%20the%20model)
Codexpedite, _“Training Flux.1 Dev LoRA with Vast.ai and FluxGym Under $1”_ [25](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training)
Hugging Face Blog, _“Diffusers welcomes FLUX-2”_ (Nov 2025) [11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22)
Reddit – r/StableDiffusion, _“Huggingface Token to access FLUX”_ (user Q&A on FLUX LoRA training)




- [1](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Creating%20consistent%20AI%20characters%20is,fast%20and%20accessible%20to%20anyone) [16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price)




- [29](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [31](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20QUANTIZATION%2C%20set%20both%20Transformer,while%20comfortably%20fitting%20the%20model)




- [25](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training)




- [11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22)




- [48](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=1)



[49](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=%E2%80%A2%20%201y%20ago)


   - GitHub – bghira/SimpleTuner README and docs [10](https://github.com/bghira/SimpleTuner#:~:text=LyCORIS%20,weights%20for%20improved%20stability%20and) [8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) (Flux model support and memory


optimization)


[1](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Creating%20consistent%20AI%20characters%20is,fast%20and%20accessible%20to%20anyone) [2](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=of%20the%20most%20reliable%20ways,fast%20and%20accessible%20to%20anyone) [3](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Image%3A%20Uploaded%20image) [15](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character) [16](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=For%20the%20final%20settings%2C%20set,balance%20of%20performance%20and%20price) [17](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=A%20well,a%20diverse%20range%20of%20shots) [18](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Boost%20Dataset%20Variety%20with%20Flux,Kontext%20Dev) [19](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=) [20](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=match%20at%20L226%20,for%20debugging%20or%20monitoring%20purposes) [51](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=Since%20we%20set%20ch3rrybl0nde%20as,so%20it%E2%80%99s%20worth%20reviewing%20them) [52](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=captions%20might%20be%20a%20bit,so%20it%E2%80%99s%20worth%20reviewing%20them) [54](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=,to%20evoke%20the%20trained%20character) [62](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=3,or%20a%20similarly%20powerful%20GPU) How to Train a Flux LoRA with FluxGym on RunPod - Next

Diffusion

[https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod)



[4](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=an%20RTX%204090%20at%20%240,is%20sufficient%20for%20budget%20training) [21](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Training%20a%20Flux,model%20without%20breaking%20the%20bank) [22](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=Key%20Benefits%20of%20Training%20Flux,Dev%20LoRA) [23](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [24](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=such%20as%20loss%2C%20to%20track,training) [25](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=2,is%20sufficient%20for%20budget%20training) [26](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar#:~:text=docker%20pull%20aidockorg%2Ffluxgym)



Training Flux.1 Dev LoRA with Vast.ai and FluxGym Under $1



[https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar](https://codexpedite.com/blog/articles/training-flux1-dev-lora-with-vastai-and-fluxgym-under-1-dollar)



[5](https://huggingface.co/blog/flux-2#:~:text=Being%20both%20a%20text,below%20or%20Ostris%27%20AI%20Toolkit) [11](https://huggingface.co/blog/flux-2#:~:text=Unfold%20to%20check%20details%20on,saving%20techniques%20used) [12](https://huggingface.co/blog/flux-2#:~:text=,by%20passing) [13](https://huggingface.co/blog/flux-2#:~:text=accelerate%20launch%20train_dreambooth_lora_flux2.py%20%5C%20,domain%22) [14](https://huggingface.co/blog/flux-2#:~:text=,ancient%20looking%20trollface%2C%20%27the%20shitposter) [35](https://huggingface.co/blog/flux-2#:~:text=LoRA%20fine) [40](https://huggingface.co/blog/flux-2#:~:text=First%2C%20instead%20of%20two%20text,known%20to%20be%20more%20beneficial) [41](https://huggingface.co/blog/flux-2#:~:text=Inference%20With%20Diffusers) [46](https://huggingface.co/blog/flux-2#:~:text=FLUX,tuning) [61](https://huggingface.co/blog/flux-2#:~:text=,bit)


[https://huggingface.co/blog/flux-2](https://huggingface.co/blog/flux-2)



Diffusers welcomes FLUX-2



[6](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=We%20first%20download%20the%20Ostris%E2%80%99,use%20any%20other%20directory%20here) [7](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5#:~:text=Step%204%3A%20Login%20to%20Hugging,Face)



How to Train a FLUX.1 LoRA for $1 | by Geronimo | Medium



[https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5](https://medium.com/@geronimo7/how-to-train-a-flux1-lora-for-1-dfd1800afce5)


[8](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) [9](https://github.com/bghira/SimpleTuner#:~:text=%2A%20TwinFlow%20Few,matching%20transformer%20with%20Chroma) [10](https://github.com/bghira/SimpleTuner#:~:text=LyCORIS%20,weights%20for%20improved%20stability%20and) [44](https://github.com/bghira/SimpleTuner#:~:text=%28guide%29%20,detect%20ComfyUI%20inputs) [45](https://github.com/bghira/SimpleTuner#:~:text=%2A%20HiDream%20MoE%20,CFG%20reintroduction%20for%20distilled%20models)

GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward image/video/audio
diffusion models.


[https://github.com/bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)


[27](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,4070%20Ti%2C%204080%2C%204090) [28](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=This%20is%20the%20first%20tier,is%20usable%20on%20well%E2%80%91tuned%20configs) [29](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [30](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20QUANTIZATION%20panel%2C%20set,float8%20%28default) [31](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20QUANTIZATION%2C%20set%20both%20Transformer,while%20comfortably%20fitting%20the%20model) [32](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=From%20available%20FLUX%20examples%20and,similar%20LoRA%20trainings) [33](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=Official%20FLUX%20LoRA%20examples%20on,a%20few%20dozen%20well%E2%80%91chosen%20images) [34](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=FLUX.2%20,batch%20sizes%20and%20latent%2Ftext%20caching) [36](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=On%20this%20tier%2C%20FLUX,tuning%20to%20avoid%20CUDA%20OOM) [37](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=streamed%20from%20CPU%20RAM%20instead,the%20GPU%20the%20whole%20time) [38](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,H100%2C%20H200%20on%20RunComfy) [39](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=) [42](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=FLUX,is%20designed%20for%201024%C3%971024%2B%20work) [53](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=,source%2C%20target%2C%20instruction%29%20triples) [55](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=this%20keeps%20quality%20high%20while,comfortably%20fitting%20the%20model) [56](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer) [57](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20TRAINING%2C%20use%20Batch%20Size,2%E2%80%99s%20fused%20transformer) [58](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=option%20if%20your%20build%20exposes,float8%20%28default) [59](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20TRAINING%20panel%2C%20keep,need%20a%20larger%20effective%20batch) [60](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=In%20the%20DATASETS%20panel%2C%20prefer,VRAM%20once%20captions%20are%20encoded) [63](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training#:~:text=You%20can%20run%20this%20FLUX,LoRA%20workflow%20in%20two%20ways) FLUX.2 [dev] LoRA Training

Guide with Ostris AI Toolkit | RunComfy

[https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-dev-lora-training)



[43](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=Just%20starting%20this%20so%20people,for%20training%20flux%20loras) [65](https://github.com/bghira/SimpleTuner/discussions/635#:~:text=)



FLUX LoRA Optimal Training · bghira SimpleTuner · Discussion #635 · GitHub



[https://github.com/bghira/SimpleTuner/discussions/635](https://github.com/bghira/SimpleTuner/discussions/635)



6


[47](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main#:~:text=By%20clicking%20,Up%20to%20review%20the)



black-forest-labs/FLUX.2-dev at main - Hugging Face



[https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main](https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main)



[48](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=1) [49](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=%E2%80%A2%20%201y%20ago) [50](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/#:~:text=Access%20to%20model%20black,be%20authenticated%20to%20access%20it)



Huggingface Token to access FLUX : r/StableDiffusion



[https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/](https://www.reddit.com/r/StableDiffusion/comments/1eusqoq/huggingface_token_to_access_flux/)



[64](https://weirdwonderfulai.art/resources/lora-training-with-ai-toolkit-on-vast-ai-under-50-cents/#:~:text=cents%20weirdwonderfulai,than%20using%20it%20on%20RunPod)



LoRA Training with AI-Toolkit on Vast.Ai under 50 cents



[https://weirdwonderfulai.art/resources/lora-training-with-ai-toolkit-on-vast-ai-under-50-cents/](https://weirdwonderfulai.art/resources/lora-training-with-ai-toolkit-on-vast-ai-under-50-cents/)


7


