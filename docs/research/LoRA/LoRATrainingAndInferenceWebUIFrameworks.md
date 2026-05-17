# **LoRA Training & Inference Web UI Frameworks**

Below are several **self-hosted web UI frameworks** that support **LoRA training** on diffusion models and
**text-to-image inference** with LoRA adapters, meeting the criteria of permissive model licensing, container
deployability, minimal filtering, and API extensibility.

## **SimpleTuner (Terminus / BGHira)**



**GitHub:** [ **bghira/SimpleTuner** – A general fine-tuning kit for diffusion models][4] (AGPL-3.0 license


[1](https://github.com/bghira/SimpleTuner#:~:text=%2A%20AGPL) ).

**Model Compatibility:** Supports a wide range of architectures including **Stable Diffusion 1.x/2.x**,
**Stable Diffusion XL**, **Stable Diffusion 3**, and new transformer-based models like **FLUX.1** (12B) and
**FLUX.2** (32B) for LoRA training [2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) . It also covers video/image models (Wan, Z-Image, etc.) – all base
models are open-source (CreativeML OpenRAIL for SD, FLUX’s own dev license, etc.).
**LoRA Training & Inference:** Provides a full training **web dashboard** to configure and run LoRA finetuning jobs. It supports LoRA and LyCORIS adapter training on all supported models (including _Flux._
_2_ ) with advanced optimizations like DeepSpeed offloading for large models [2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) . During training, you
can attach LoRA adapters for **validation image generation** without altering the model (to preview
results) [3](https://github.com/bghira/SimpleTuner#:~:text=%2A%20Loss%20functions%20,details) . After training, the resulting LoRA can be downloaded and used for text-to-image
generation in any Stable Diffusion pipeline (SimpleTuner itself can run inference via its







[1](https://github.com/bghira/SimpleTuner#:~:text=%2A%20AGPL)







[2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B)







[2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B)



[3](https://github.com/bghira/SimpleTuner#:~:text=%2A%20Loss%20functions%20,details)



`predict.py`, or you can load the LoRA into your preferred UI).



**Licensing & Open Models:** The SimpleTuner code is AGPL-3.0 [1](https://github.com/bghira/SimpleTuner#:~:text=%2A%20AGPL), but it **uses only permissively**
**licensed models** – e.g. Stable Diffusion (CreativeML OpenRAIL-M), FLUX (open dev license), etc. No
proprietary or closed models are required.
**Content Filtering: No baked-in filtering.** SimpleTuner does not impose any safety filters on
training data or outputs. The user has full control of prompts and dataset. (Since it’s self-hosted, any
content-safety or content restrictions would only come from the chosen model; e.g. Stable Diffusion’s default
safety checker is not used here.)
**Deployment: Docker support** is built-in (provides a Dockerfile, docker-compose, etc. [4](https://github.com/bghira/SimpleTuner#:~:text=match%20at%20L321%20docker) [5](https://github.com/bghira/SimpleTuner#:~:text=docker) ). You
can run the web UI on a local machine or deploy it to cloud GPU instances. While not officially
providing one-click RunPod templates, the community has run it in containers on RunPod/Vast – its
multi-worker design even supports distributed training across ephemeral cloud workers [6](https://github.com/bghira/SimpleTuner#:~:text=SimpleTuner%20includes%20a%20complete%20multi,features%E2%80%94free%20and%20open%20source%2C%20forever) [7](https://github.com/bghira/SimpleTuner#:~:text=to%20restrict%20configs%2C%20hardware%2C%20or,waiting%20jobs%2C%20and) .
**API Extensibility:** SimpleTuner is designed as a multi-user training server with an API-layer for job
scheduling. It features **REST API keys** and multi-tenant orchestration [8](https://github.com/bghira/SimpleTuner#:~:text=%2A%20SSO%20Integration%20,daily%2Fmonthly%29%2C%20job) [9](https://github.com/bghira/SimpleTuner#:~:text=share%20scheduling%20across%20teams%2C%20starvation,limited%20scope%20for%20CI%2FCD%20pipelines), meaning you can
programmatically launch training runs or integrate it into pipelines. The UI itself is built on a
backend that could be invoked via HTTP or CLI (although the official focus is the web dashboard). In
short, it’s **highly extensible** – users can script training tasks or use webhooks, and no external
restrictions are imposed on model or prompt usage.




- [1](https://github.com/bghira/SimpleTuner#:~:text=%2A%20AGPL)








- [4](https://github.com/bghira/SimpleTuner#:~:text=match%20at%20L321%20docker) [5](https://github.com/bghira/SimpleTuner#:~:text=docker)



[6](https://github.com/bghira/SimpleTuner#:~:text=SimpleTuner%20includes%20a%20complete%20multi,features%E2%80%94free%20and%20open%20source%2C%20forever) [7](https://github.com/bghira/SimpleTuner#:~:text=to%20restrict%20configs%2C%20hardware%2C%20or,waiting%20jobs%2C%20and)







[8](https://github.com/bghira/SimpleTuner#:~:text=%2A%20SSO%20Integration%20,daily%2Fmonthly%29%2C%20job) [9](https://github.com/bghira/SimpleTuner#:~:text=share%20scheduling%20across%20teams%2C%20starvation,limited%20scope%20for%20CI%2FCD%20pipelines)


## **Kohya’s Stable Diffusion GUI**



**GitHub:** [ **bmaltais/kohya_ss** – Kohya’s GUI for Stable Diffusion training scripts][7] (Apache-2.0
licensed [10](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20README%20%2A%20Apache,Security) ).







[10](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20README%20%2A%20Apache,Security)



1


**Model Compatibility:** Geared for **Stable Diffusion family models** – supports **SD 1.x, SD 2.x**, and
**SDXL** fine-tuning. It can train **LoRA** as well as full-model DreamBooth and Textual Inversion,
including support for **SDXL LoRA training** [11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation) . (It uses Kohya’s backend scripts, which currently
target the CompVis Stable Diffusion architectures; emerging architectures like FLUX are **not yet**
**supported** here.) Base models are loaded from local checkpoints (e.g.







[11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation)



`v1-5-pruned.safetensors` under OpenRAIL-M, SDXL under OpenRAIL), or any custom model



path, so you’re using permissively licensed weights of your choice.
**LoRA Training & Inference:** Provides a **user-friendly Gradio web UI** for configuring training
parameters and launching jobs [11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation) . LoRA training is fully supported via a dedicated tab – you can set
the dataset, parameters (rank, learning rate, epochs, etc.), and train LoRAs for your model. The tool
also supports _sample image generation during training_ to monitor progress [12](https://github.com/bmaltais/kohya_ss#:~:text=,GPU%20Utilization%20Issue) . After training, you’d
use another tool for inference (Kohya’s GUI will output the `.safetensors` LoRA file and some

sample images, but it’s not an all-purpose image generator UI). However, since it’s built for fine
tuning, **no filtering or safety filters** are applied to outputs – you have direct control over training
and sample prompts.
**Licensing:** The GUI code is Apache-2.0 [10](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20README%20%2A%20Apache,Security) and relies on open-source training scripts. The models
you train on must be accessible (for example, Stable Diffusion 1.5 is CreativeML OpenRAIL-M). There
are **no restrictive fine-tuning rules** enforced by the tool itself – it will train whatever you give it.
**Content Filtering: Unfiltered by default.** Kohya’s GUI does not include any hard-coded content
moderation. Because it’s an offline tool, it assumes you manage your own data responsibly. Even the
Stable Diffusion safety checker (content-safety filter) is typically disabled for training and sample generation.
(Users have reported that if using diffusers pipelines for sample preview, the `--no-content_safety_checker`

flag can be set, and by default the safety checker is off in recent versions [13](https://github.com/invoke-ai/InvokeAI/issues/1729#:~:text=,checker%20should%20be%20turned%20off) .) In practice, it’s known
as an unfiltered training solution.
**Deployment:** Supports **Docker and cloud deployment** . The project includes a Dockerfile and even a
one-click **RunPod** setup script [14](https://github.com/bmaltais/kohya_ss#:~:text=,user%2C%20Docker%20is%20also%20supported) . Many users run this GUI on cloud GPU services (RunPod, Vast.ai,
etc.) since it’s convenient for training on rented GPUs. It can also be installed locally via pip or `uv`







[11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation)






- [10](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20README%20%2A%20Apache,Security)







[13](https://github.com/invoke-ai/InvokeAI/issues/1729#:~:text=,checker%20should%20be%20turned%20off)







[14](https://github.com/bmaltais/kohya_ss#:~:text=,user%2C%20Docker%20is%20also%20supported)



virtual environment on Windows/Linux, or run on Colab [15](https://github.com/bmaltais/kohya_ss#:~:text=You%20can%20run%20,solutions%20like%20Colab%20or%20Runpod) . There are community images and
scripts to quickly launch it on cloud instances (for example, RunPod has a community template for
Kohya’s trainer).
**API Extensibility: No official external API** – Kohya’s GUI is primarily an interactive tool. However, it
automatically generates the underlying **CLI commands** for Kohya’s `train_network.py` etc., so


advanced users can copy those and automate training outside the UI [16](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20Easy,tuning%2C%20and%20SDXL%20training) . The GUI doesn’t expose a

documented REST API for external calls. If needed, you could run the training scripts directly (since
they are open source) or modify the Gradio app. But out-of-the-box, think of this as a standalone
web UI (self-hosted) rather than a persistent service to be called by other applications.



[15](https://github.com/bmaltais/kohya_ss#:~:text=You%20can%20run%20,solutions%20like%20Colab%20or%20Runpod)








## **AUTOMATIC1111 Stable Diffusion Web UI (with LoRA Extensions)**



**GitHub:** [ **AUTOMATIC1111/stable-diffusion-webui** – Stable Diffusion Web UI][22] (AGPL-3.0 license


[17](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt#:~:text=AUTOMATIC1111%2Fstable,strongest%20copyleft%20license%20are) ).

**Model Compatibility:** Supports **Stable Diffusion v1.x and v2.x models** by default, and has
community support for **SDXL** as well. You can load any custom checkpoint or SafeTensors (so long as
it’s a standard Stable Diffusion model). **LoRA compatibility for inference** is built-in – the UI can
apply LoRA weights on the fly to any loaded model (e.g. `.safetensors` LoRAs from CivitAI).







[17](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt#:~:text=AUTOMATIC1111%2Fstable,strongest%20copyleft%20license%20are)







However, it **does not natively support non-standard architectures like FLUX** (the UI is tightly
coupled to Stable Diffusion’s U-Net/Clip structure). For those models, other UIs (ComfyUI or


2


specialized forks) are needed. All popular base models it uses (SD 1.5, 2.1, SDXL) are permissively
licensed (CreativeML OpenRAIL).

- **LoRA Training & Inference:** This UI’s strength is **text-to-image inference** – it’s the de-facto

interface for generating images with Stable Diffusion. Out-of-the-box, it did **not include training** .
But the community developed extensions to train custom models and LoRAs. For example, the builtin **Dreambooth** extension can fine-tune models or produce LoRAs with minimal VRAM, and there’s a
dedicated **LoRA training extension** ( `sd-webui-traintrain` ) that integrates a training tab into


the UI [18](https://github.com/hako-mikan/sd-webui-traintrain#:~:text=hako,Notifications%20You%20must%20be) . Using these, you can train LoRAs on your data directly from the Web UI (the extensions
internally call libraries like Diffusers or Kohya scripts). After training, the WebUI can immediately
**load the new LoRA** and generate images with it, in the same interface. Inference with LoRAs is
trivial: just put the LoRA file in the `models/Lora` folder and select it in the UI to apply on any



prompt.
**Licensing:** The WebUI is AGPL-3.0 [17](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt#:~:text=AUTOMATIC1111%2Fstable,strongest%20copyleft%20license%20are) and completely open source. It lets you choose any model – by
default no model is included, and you download your own. The typical choices (Stable Diffusion
checkpoints) are all open-license for research/commercial use under conditions (OpenRAIL-M). No
closed or propietary model is enforced.
**Content Filtering: No hardcoded filtering.** Automatic1111’s UI is known for being **unfiltered** –
it does **not** include the Stable Diffusion “safety checker” unless you add it. In fact, users often discuss
how _the base UI has no content-safety filter_ (only optional add-ons) [19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful) . There is an **content-safety filter extension**
available [19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful), but by default the UI will happily generate any content the model allows. This was a
key reason this WebUI became popular, as it imposes no additional content restrictions beyond the
model’s own biases. Prompting and fine-tuning are unrestricted (aside from obvious legal/commonsense limits left to the user).

**Deployment:** Extremely popular for self-hosting – numerous **Docker images and cloud templates**
exist. While the official repo doesn’t ship a Docker, community-maintained images on Docker Hub
(e.g. `automatic1111-webui` ), as well as one-click deploy buttons for RunPod and Vast, are readily

available. For example, RunPod’s Stable Diffusion template uses this UI (with an API enabled) and
can launch it on a GPU with one click. The UI runs as a Gradio app on port 7860 by default, which you
can expose publicly or restrict. Many users also run it in Colab notebooks. In short, **containerization**
**and cloud hosting are well-trodden** for A1111’s UI.
**API Extensibility: Yes – REST API available.** The WebUI can be launched with an `--api` flag which




- The WebUI is AGPL-3.0 [17](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt#:~:text=AUTOMATIC1111%2Fstable,strongest%20copyleft%20license%20are)







[19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful)



[19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful)







available. For example, RunPod’s Stable Diffusion template uses this UI (with an API enabled) and
can launch it on a GPU with one click. The UI runs as a Gradio app on port 7860 by default, which you
can expose publicly or restrict. Many users also run it in Colab notebooks. In short, **containerization**
**and cloud hosting are well-trodden** for A1111’s UI.

- **API Extensibility: Yes – REST API available.** The WebUI can be launched with an `--api` flag which



exposes a RESTful API (Swagger spec provided at `http://host:7860/docs` ) for text-to-image,


image-to-image, etc. This makes it easy to integrate into other applications or automation scripts.
Additionally, its **extension system** means you can add new endpoints or features. Many third-party
tools (like Adobe Photoshop plugins, Blender addons, etc.) interface with Automatic1111’s API to
generate images. The combination of a powerful UI and an open API gives you flexible integration
options. (Do note that the API inherits the UI’s lack of filtering – e.g. it will return content-safety images if
your model/prompt produce them, unless you manually add a filter extension.)

## **InvokeAI (InvokeAI 3.x “Creative Engine”)**



**GitHub:** [ **invoke-ai/InvokeAI** – InvokeAI Stable Diffusion Toolkit][10] (MIT License [20](https://www.rundiffusion.com/software-licensing#:~:text=Open%20Source%20Licenses%20%C2%B7%20Automatic1111,MIT%20License) ).
**Model Compatibility:** InvokeAI supports **Stable Diffusion 1.0 through 2.x**, **Stable Diffusion XL**,
and related diffusers-based models. It originally launched with stable diffusion 1.4/1.5 and has kept
up with SDXL in version 3.0+ [21](https://www.youtube.com/watch?v=8cVnooYgpDc#:~:text=InvokeAI%203.0%20,fuctions%20and%20optimized%20their%20UI) . You can import any model checkpoint or HuggingFace model (so
long as it’s Diffusers or CompVis compatible). By default, it uses open models (the installer can
download SD 1.5, which is CreativeML OpenRAIL-M, or SDXL under the same license). **LoRA** is fully




- **GitHub:** [ **invoke-ai/InvokeAI** [20](https://www.rundiffusion.com/software-licensing#:~:text=Open%20Source%20Licenses%20%C2%B7%20Automatic1111,MIT%20License)







[21](https://www.youtube.com/watch?v=8cVnooYgpDc#:~:text=InvokeAI%203.0%20,fuctions%20and%20optimized%20their%20UI)



3


supported on the **inference side** as of v3 – you can add LoRA models to influence generation, and it
supports things like character LoRAs and ControlNet in its workflow editor [22](https://www.youtube.com/watch?v=1Iz4F7o6hgQ#:~:text=Newly%20Released%20Invoke%20AI%203,Get%20Invoke%20AI%20here) . For **training**,
InvokeAI has a separate library and UI components (InvokeAI Training) for fine-tuning SD models
and LoRAs. It mainly targets stable diffusion models (there’s no native support for Flux or non-SD
architectures yet in the public release).
**LoRA Training & Inference: Text-to-image UI:** InvokeAI offers a polished web interface for
generating images, with a **node-based Workflow Editor** and traditional prompt-based GUI. This
interface allows applying LoRAs in generation and chaining various effects. **LoRA Training:** In
version 3.x, InvokeAI introduced a dedicated training panel (powered by the external `invoke-`



[22](https://www.youtube.com/watch?v=1Iz4F7o6hgQ#:~:text=Newly%20Released%20Invoke%20AI%203,Get%20Invoke%20AI%20here)







`training` library) for common fine-tuning tasks [23](https://invoke-ai.github.io/InvokeAI/#:~:text=Invoke%20Invoke%20Training%20has%20moved,can%20find%20more%20by) . Users can train Textual Inversions or LoRAs



from within the InvokeAI environment – e.g. by providing a dataset and configuration, then running
a training workflow. This leverages the same underlying scripts as Kohya or Diffusers but wrapped in
InvokeAI’s UI. (It’s not as one-click simple as SimpleTuner, but it’s integrated.) After training, the new
LoRA can be loaded into the workflow editor to generate images with the trained style/subject.
Essentially, InvokeAI is evolving into a **one-stop shop** for both generation and moderate training.
**Licensing:** The core InvokeAI code is MIT licensed [20](https://www.rundiffusion.com/software-licensing#:~:text=Open%20Source%20Licenses%20%C2%B7%20Automatic1111,MIT%20License) (very permissive). It uses Stable Diffusion
weights (which are permissively released for commercial use with conditions). InvokeAI’s philosophy
is to be **commercial-friendly** and open source. Models included or downloaded are all open licensed
(e.g., they avoid any model with non-commercial-only licenses by default, unless you choose to load

one).

**Content Filtering: Optional safety, user-controlled.** InvokeAI historically included an **content-safety**
**checker** (the Diffusers safety filter) that could be toggled. In recent versions, the default is _no_
_filtering_ unless you enable it. For example, in InvokeAI 2.2 the `--no-content_safety_checker` flag was

introduced and if you run normally, the safety checker is off by default [13](https://github.com/invoke-ai/InvokeAI/issues/1729#:~:text=,checker%20should%20be%20turned%20off) . There have been user
reports of needing to disable the content-safety filter in settings, but generally InvokeAI does not filter
outputs beyond what the user configures. The UI does not have hardcoded forbidden terms or
anything – you have full control to generate any content that your loaded model allows. (If using
their official installer with SDXL, you might get the default SDXL safety guidance, but it can be turned
off easily.) Overall, InvokeAI is considered **unfiltered** – aligned with the open-source ethos that the

user decides how to use it.

**Deployment: Docker & Cloud:** InvokeAI can be installed via pip/conda, and community-provided
Docker images exist. The official docs detail how to run it on various platforms, and it’s known to
work on Windows, Linux, Mac (even Apple Silicon via Torch MPS). While there might not be an
“official” one-click RunPod template from InvokeAI maintainers, **third-party templates** are available
(e.g. some RunPod community scripts for InvokeAI WebGUI). Version 3’s node-based GUI is heavier
than A1111, but it’s still feasible to host. If not using Docker, InvokeAI has its own launcher script that
sets up a local web server. For cloud, one can use services like Vast.ai or runpod by simply launching
the container or the installer on a machine – many have done so for persistent InvokeAI instances.
**API Extensibility: Partially – internal API (not fully public).** InvokeAI’s web server does have an
API that the frontend uses (for example, to queue generation tasks), but it’s not thoroughly
documented for third-party use [24](https://github.com/invoke-ai/InvokeAI/issues/6581#:~:text=,fully%20documented%20except%20in%20code) . An issue from the devs notes that the REST API exists but is
meant for the UI client, and external use would require reading the source (no stable external
endpoints guaranteed) [24](https://github.com/invoke-ai/InvokeAI/issues/6581#:~:text=,fully%20documented%20except%20in%20code) . That said, advanced users can script against InvokeAI by using its
**Python API** (you can import the `invokeai` library and call generation functions directly in a Python




- The core InvokeAI code is MIT licensed [20](https://www.rundiffusion.com/software-licensing#:~:text=Open%20Source%20Licenses%20%C2%B7%20Automatic1111,MIT%20License)

















[24](https://github.com/invoke-ai/InvokeAI/issues/6581#:~:text=,fully%20documented%20except%20in%20code)



[24](https://github.com/invoke-ai/InvokeAI/issues/6581#:~:text=,fully%20documented%20except%20in%20code)



context [25](https://www.reddit.com/r/invokeai/comments/1edivme/api_texttoimage_example/#:~:text=API%20text,API) ). There is also interest in an Automatic1111-compatible API mode [26](https://github.com/invoke-ai/InvokeAI/issues/2205#:~:text=,fork%20of%20InvokeAI%20but), but as of now it’s
not plug-and-play. In summary, you _can_ extend or wrap InvokeAI – e.g. integrate the `invoke-`

`training` or use the workflow engine programmatically – but expect to do some custom coding. It


4


doesn’t have a simple REST endpoint for “generate image” out-of-the-box like A1111, aside from the
community forks or the possibility of using the built-in API with careful reverse-engineering. The
upside is that the **workflow editor** lets you design complex pipelines (which could be saved and run
headlessly), so one could deploy an InvokeAI workflow as a service with some effort.

## **ComfyUI (Node-Based UI + LoRA Extension)**



**GitHub:** [ **comfyanonymous/ComfyUI** – ComfyUI node editor][28] (GPLv3 license). **LoRA Training**
**Extension:** [ **shootthesound/comfyUI-Realtime-Lora** – real-time LoRA trainer nodes][17] (MIT

licensed extension).
**Model Compatibility: Extremely modular.** ComfyUI can load and operate on any diffusion model
supported by **HuggingFace Diffusers** or custom checkpoints. Natively it’s used for Stable Diffusion
1.x/2.x and SDXL (OpenRAIL models), but it’s also extensible to others. With the appropriate custom
nodes, ComfyUI can run **FLUX**, **Zelda/Z-image**, **Waifu, Kandinsky**, etc. The **Realtime LoRA**
**extension** specifically adds support for training LoRAs on **FLUX.1-dev, WAN 2.2, Z-Image Turbo,**
**SDXL, SD1.5** and more [27](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=Train%20and%20block%20edit%20and,5) . In short, ComfyUI itself is model-agnostic (if you have a pipeline or
checkpoint, you can create a workflow for it). All these models are open-license (e.g. FLUX dev is free
for non-commercial use, SD is OpenRAIL).
**LoRA Training & Inference: Inference:** ComfyUI is a powerful **graphical workflow builder** for
diffusion inference. You add nodes for loading models, applying LoRAs, textual inversion
embeddings, doing text-to-image, image-to-image, etc. It has no built-in LoRA training, but that’s
where the **“Realtime LoRA Trainer” extension** comes in [28](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Basically%20you%20pass%20it%20images,part%20of%20my%20own%20workflow) [29](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=EDIT%20,Lora) . This extension provides special
ComfyUI nodes that interface with external training libraries (like _AI-Toolkit_ [30](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=fly%2C%20using%20your%20local%20install,part%20of%20my%20own%20workflow) and _Kohya’s sd-scripts_


[31](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=SD%20Scripts%20%28for%20SDXL%29%3A%20https%3A%2F%2Fgithub.com%2Fkohya)

) to train LoRAs inside the ComfyUI workflow. Practically, you can construct a graph where you
feed in a folder of training images, click “execute”, and the nodes will launch a LoRA training job (for
say FLUX or SDXL) and then automatically use the resulting LoRA in subsequent generation nodes –
**all within one ComfyUI session** [32](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Image%3A%20r%2FStableDiffusion%20,image%2FWan%2FFlux%20Dev) . This is very cutting-edge: e.g. you can generate an image, then
decide to fine-tune a LoRA on-the-fly using that image (or a small dataset), then apply it immediately
to influence the next generation. The extension supports saving the LoRA to file and even interactive
_block-wise merging/editing_ of LoRA weights in real-time. After training, ComfyUI (with a standard
LoRA loader node) can apply the new adapter to any prompt. Essentially, ComfyUI + this extension
offers **fully modular training and inference** in one interface, with support for experimental models

(Flux, etc.).

**Licensing:** ComfyUI is GPL v3 (open source), and the Real-time LoRA extension is MIT [33](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=License) . All
underlying training code it calls (AI-Toolkit by Ostris, Kohya scripts) are MIT or Apache licensed, and
models used are open. There is **no enforced fine-tuning restriction** – you can train on any data and
any model you have rights to.
**Content Filtering: None.** ComfyUI has zero built-in filtering – it’s a low-level tool that doesn’t
assume anything about content. It will happily generate content-safety or any material the model can
produce. Since workflows are user-defined, you could even insert your own filter node if desired, but
by default **nothing is filtered or blocked** . The LoRA training extension likewise imposes no safety
checks (other than what the model might inherently do or what the user chooses to include in a
training workflow). This framework is considered as _“raw”_ as it gets – maximal freedom, but it
expects the user to know what they are doing.
**Deployment:** ComfyUI is a lightweight Python app and can run in Docker or virtually anywhere.
While not as popular as A1111 for plug-and-play containers, **community Docker images exist** .
Users have deployed ComfyUI on RunPod and vast.ai – for instance, by using the `nvidia/cuda`











[27](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=Train%20and%20block%20edit%20and,5)







[28](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Basically%20you%20pass%20it%20images,part%20of%20my%20own%20workflow) [29](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=EDIT%20,Lora)



[30](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=fly%2C%20using%20your%20local%20install,part%20of%20my%20own%20workflow)



[31](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=SD%20Scripts%20%28for%20SDXL%29%3A%20https%3A%2F%2Fgithub.com%2Fkohya)



[32](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Image%3A%20r%2FStableDiffusion%20,image%2FWan%2FFlux%20Dev)




- [33](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=License)











5


base and running `pip install comfyui` or similar. The RealTime LoRA extension would just be


added to the `custom_nodes` folder. Because ComfyUI’s GUI is web-based (running on e.g.


`127.0.0.1:8188` ), hosting it on a remote GPU and accessing via browser is straightforward. There


are also hosted solutions (like ComfyUI on Modal or Baseten) showing how to serve it as a persistent
API [34](https://modal.com/docs/examples/comfyapp#:~:text=Run%20Flux%20on%20ComfyUI%20as,into%20a%20scalable%20API%20endpoint) . In summary, ComfyUI is **easily containerized** (no complex dependencies beyond PyTorch)
and often used in cloud GPU notebooks. The extension doesn’t change that – it relies on standard
training libs that will be installed alongside.
**API Extensibility: Yes.** ComfyUI was designed with an API in mind. It provides a **REST API** to execute
workflows by sending JSON payloads describing the graph and returning results [35](https://comfy.icu/docs/api#:~:text=Run%20ComfyUI%20with%20an%20API,on%20maintaining%20own%20GPU%20infrastructure) . For example,
you can POST a workflow graph to a ComfyUI server and get back an image or latent. The official
docs detail endpoints for listing nodes, uploading custom nodes, and running workflows [36](https://docs.comfy.org/registry/api-reference/overview#:~:text=API%20Overview%20%C2%B7%20%E2%80%8B,Was%20this%20page%20helpful) . This
means you can use ComfyUI headless as a diffusion **microservice** . Given this, one could also
integrate LoRA training in an API call – for instance, send a workflow that includes the LoRA training
nodes to the API server. (This might be advanced, but it’s feasible since the extension’s nodes are just
another part of the graph.) Additionally, ComfyUI’s **modular architecture** allows wrapping or
embedding it – some projects use ComfyUI as a backend engine behind their own interface. The
bottom line: ComfyUI is extremely extensible, both via custom nodes (like the LoRA trainer) and via
API usage. It’s a great choice if you need a **fully customizable, code-friendly** solution without any

hidden limits.



[34](https://modal.com/docs/examples/comfyapp#:~:text=Run%20Flux%20on%20ComfyUI%20as,into%20a%20scalable%20API%20endpoint)







[35](https://comfy.icu/docs/api#:~:text=Run%20ComfyUI%20with%20an%20API,on%20maintaining%20own%20GPU%20infrastructure)



[36](https://docs.comfy.org/registry/api-reference/overview#:~:text=API%20Overview%20%C2%B7%20%E2%80%8B,Was%20this%20page%20helpful)


## **FluxGym (Flux LoRA Trainer UI)**



**GitHub:** [ **cocktailpeanut/fluxgym** – “Flux Gym” LoRA training UI][19] (MIT License [37](https://github.com/cocktailpeanut/fluxgym#:~:text=) ).
**Model Compatibility:** Initially built for **FLUX.1** (dev) model LoRA training, with special focus on _low-_
_VRAM GPUs_ . By default it supports FLUX.1 and variants (Flux Schnell, etc.) [38](https://github.com/cocktailpeanut/fluxgym#:~:text=Supported%20Models) . However, FluxGym is
**extensible to other models** – it uses _Kohya’s training scripts_ under the hood, so you can add any
Stable Diffusion or Diffusers model to its config. In fact, an update in September 2025 added support
for **custom base models** beyond Flux (simply by editing a YAML file) [39](https://github.com/cocktailpeanut/fluxgym#:~:text=,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638) . This means you could train
LoRAs for Stable Diffusion 1.5, SDXL, or others using the same interface (though the UI may not have
pre-populated fields for them). All models assumed are open-license (Flux’s weights are free for
research; if you use SD, it’s OpenRAIL, etc.).
**LoRA Training & Inference: Training UI:** FluxGym provides a super simple **Gradio web UI** to

configure LoRA training jobs for diffusion models [40](https://github.com/cocktailpeanut/fluxgym#:~:text=Dead%20simple%20web%20UI%20for,12GB%2F16GB%2F20GB%29%20support) . It presents basic options (dataset path,
learning rate, epochs, etc.) with an “Advanced” tab exposing the full range of Kohya’s sd-scripts
options [41](https://github.com/cocktailpeanut/fluxgym#:~:text=,script%20powered%20by%20Kohya%20Scripts) . For example, you can fine-tune a FLUX.1 dev checkpoint on a custom image set to
produce a LoRA. It was specifically optimized to allow FLUX training on **12GB–20GB VRAM GPUs** by
using 8-bit and 2-bit quantization techniques [42](https://www.reddit.com/r/StableDiffusion/comments/1ekb9qj/simpletuner_v098_quantised_flux_training_in_40/#:~:text=SimpleTuner%20v0,potato%20LoRA%20of%20your%20dreams) [43](https://github.com/cocktailpeanut/fluxgym#:~:text=1,it%20uses%20Kohya%20Scripts%20underneath) . **Inference:** FluxGym will **automatically**
**generate sample images** using the trained LoRA after each training session (this was added in a
recent update [44](https://github.com/cocktailpeanut/fluxgym#:~:text=%2A%20September%2016%3A%20Added%20,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638) ). That is, once training completes, it can apply the LoRA to the base model and
output a few images for you to review quality. However, it’s not a full interactive generation UI – for
inference beyond the automatic samples, you would load the LoRA into another tool (like ComfyUI or
A1111) or write a small diffusers script. FluxGym’s primary role is making LoRA training easy, rather
than serving as a general-purpose image generator.
**Licensing:** The UI is MIT licensed [37](https://github.com/cocktailpeanut/fluxgym#:~:text=) and combines two open components: the frontend is adapted
from Ostris’s **AI-Toolkit UI** (which is MIT) and the backend uses **Kohya’s sd-scripts** (also opensource) [41](https://github.com/cocktailpeanut/fluxgym#:~:text=,script%20powered%20by%20Kohya%20Scripts) [45](https://github.com/cocktailpeanut/fluxgym#:~:text=2.%20The%20AI,it%20uses%20Kohya%20Scripts%20underneath) . As such, there are no license encumbrances. The base _FLUX.1-dev_ model it targets is
under a non-commercial research license [46](https://huggingface.co/XLabs-AI/flux-lora-collection#:~:text=Text,generation%20%20%20%2016), meaning you shouldn’t use Flux outputs commercially




- **GitHub:** [ [37](https://github.com/cocktailpeanut/fluxgym#:~:text=)







[38](https://github.com/cocktailpeanut/fluxgym#:~:text=Supported%20Models)



[39](https://github.com/cocktailpeanut/fluxgym#:~:text=,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638)







[40](https://github.com/cocktailpeanut/fluxgym#:~:text=Dead%20simple%20web%20UI%20for,12GB%2F16GB%2F20GB%29%20support)



[41](https://github.com/cocktailpeanut/fluxgym#:~:text=,script%20powered%20by%20Kohya%20Scripts)



[42](https://www.reddit.com/r/StableDiffusion/comments/1ekb9qj/simpletuner_v098_quantised_flux_training_in_40/#:~:text=SimpleTuner%20v0,potato%20LoRA%20of%20your%20dreams) [43](https://github.com/cocktailpeanut/fluxgym#:~:text=1,it%20uses%20Kohya%20Scripts%20underneath)



[44](https://github.com/cocktailpeanut/fluxgym#:~:text=%2A%20September%2016%3A%20Added%20,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638)




- The UI is MIT licensed [37](https://github.com/cocktailpeanut/fluxgym#:~:text=)



[41](https://github.com/cocktailpeanut/fluxgym#:~:text=,script%20powered%20by%20Kohya%20Scripts) [45](https://github.com/cocktailpeanut/fluxgym#:~:text=2.%20The%20AI,it%20uses%20Kohya%20Scripts%20underneath)



[46](https://huggingface.co/XLabs-AI/flux-lora-collection#:~:text=Text,generation%20%20%20%2016)



6


without a commercial license – but that is a model restriction, not something enforced by the UI. If
you swap in a model like Stable Diffusion 1.5, you abide by that model’s OpenRAIL terms (which
allow commercial use with some rules). FluxGym itself puts **no additional restrictions** on finetuning or outputs.
**Content Filtering: No content filters.** FluxGym does not include any content-safety or profanity filters. It will
train on whatever images/captions you feed it. Since it’s focused on training, the only “outputs” are
the LoRA file and some sample images – none of which are subject to any automatic moderation.
The design goal is simplicity, so it doesn’t impose any safety layers (you have full responsibility for
the training data’s content). Generated samples too come straight from the model without any
intervention. In short, **unfiltered by design** .
**Deployment: Container & Cloud:** FluxGym provides a Dockerfile and even a one-click installer via a
tool called Pinokio [47](https://github.com/cocktailpeanut/fluxgym#:~:text=https%3A%2F%2Fpinokio) . It’s meant to be very easy to run anywhere. Docker support was announced
as of Sept 25, 2025 [48](https://github.com/cocktailpeanut/fluxgym#:~:text=News), which also enabled automatic downloading of base models. You can fire up
`docker-compose.yml` and get the UI running locally or on a server. Because it’s Gradio-based, you













can port-forward the interface or use services like **RunPod** . In fact, there are guides on using
FluxGym on RunPod [49](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=How%20to%20Train%20a%20Flux,UI%20and%20begin%20the) – demonstrating how to quickly spin up a cloud GPU and train a Flux LoRA
via the web UI. Similarly, Vast.ai users have containerized it for on-demand training. The small
footprint (only ~10 GB image including model) and low VRAM requirements make it **accessible on**

**modest cloud instances** .

**API Extensibility: Limited (UI-focused).** FluxGym itself doesn’t expose a documented API for
external calls – it’s intended as a simple web app. Under the hood, though, it’s calling Kohya’s
training CLI. If one wanted, they could bypass the UI and call those scripts directly (the UI essentially
forms a command and executes it). Since the UI is Gradio, you _could_ call its functions via Python if
you imported it as a module, but that’s unconventional. The likely use-case is interactive usage. That
said, because it is open source, one could modify it or wrap it: e.g., you could run FluxGym on a
server and script interactions (like using Selenium or Gradio’s API endpoints) to start a training and
monitor progress. This is not a built-in capability, however. In summary, **FluxGym is meant for**
**human-in-the-loop operation** . For automation, it might be better to directly use the training library
(Kohya scripts or Ostris’s AI-Toolkit) which it’s built on. Nonetheless, the existence of ready Docker
and simple UI means you can fairly easily plug it into a larger workflow manually.



[49](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=How%20to%20Train%20a%20Flux,UI%20and%20begin%20the)







**References:** The information above is drawn from the official documentation and repositories of each
project, as well as user guides and community discussions. Key sources include project README files (for
features, license info, and model support) and known issues or forum posts regarding content filtering and


[2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B)

deployment options. For example, SimpleTuner’s README confirms its multi-model LoRA support,
Kohya’s GUI documentation notes its training features including LoRA and SDXL [11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation), and community Q&As
highlight the lack of forced filters in these UIs (e.g. Automatic1111 requiring an extension for content-safety filtering


[19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful)

). We have cited these sources inline to provide verifiable details for each framework. Each of these tools
is under active open-source development as of 2025, so capabilities may expand further (e.g. InvokeAI
integrating more training features, ComfyUI nodes for new models, etc.), but they all currently satisfy the
criteria of LoRA training + inference with permissive models and minimal restrictions.



[2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B)



[11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation)



[19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful)



7


[1](https://github.com/bghira/SimpleTuner#:~:text=%2A%20AGPL) [2](https://github.com/bghira/SimpleTuner#:~:text=Stable%20Diffusion%20XL%203,2B) [3](https://github.com/bghira/SimpleTuner#:~:text=%2A%20Loss%20functions%20,details) [4](https://github.com/bghira/SimpleTuner#:~:text=match%20at%20L321%20docker) [5](https://github.com/bghira/SimpleTuner#:~:text=docker) [6](https://github.com/bghira/SimpleTuner#:~:text=SimpleTuner%20includes%20a%20complete%20multi,features%E2%80%94free%20and%20open%20source%2C%20forever) [7](https://github.com/bghira/SimpleTuner#:~:text=to%20restrict%20configs%2C%20hardware%2C%20or,waiting%20jobs%2C%20and) [8](https://github.com/bghira/SimpleTuner#:~:text=%2A%20SSO%20Integration%20,daily%2Fmonthly%29%2C%20job) [9](https://github.com/bghira/SimpleTuner#:~:text=share%20scheduling%20across%20teams%2C%20starvation,limited%20scope%20for%20CI%2FCD%20pipelines)

GitHub - bghira/SimpleTuner: A general fine-tuning kit geared toward
image/video/audio diffusion models.


[https://github.com/bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)



[10](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20README%20%2A%20Apache,Security) [11](https://github.com/bmaltais/kohya_ss#:~:text=This%20project%20provides%20a%20user,Rank%20Adaptation) [12](https://github.com/bmaltais/kohya_ss#:~:text=,GPU%20Utilization%20Issue) [14](https://github.com/bmaltais/kohya_ss#:~:text=,user%2C%20Docker%20is%20also%20supported) [15](https://github.com/bmaltais/kohya_ss#:~:text=You%20can%20run%20,solutions%20like%20Colab%20or%20Runpod) [16](https://github.com/bmaltais/kohya_ss#:~:text=%2A%20Easy,tuning%2C%20and%20SDXL%20training)



GitHub - bmaltais/kohya_ss



[https://github.com/bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)



[13](https://github.com/invoke-ai/InvokeAI/issues/1729#:~:text=,checker%20should%20be%20turned%20off)




[bug]: the content-safety checker isn't being disabled · Issue #1729 - GitHub



[https://github.com/invoke-ai/InvokeAI/issues/1729](https://github.com/invoke-ai/InvokeAI/issues/1729)



[17](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt#:~:text=AUTOMATIC1111%2Fstable,strongest%20copyleft%20license%20are)



stable-diffusion-webui/LICENSE.txt at master - GitHub



[https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/LICENSE.txt)



[18](https://github.com/hako-mikan/sd-webui-traintrain#:~:text=hako,Notifications%20You%20must%20be)



hako-mikan/sd-webui-traintrain: LoRA training extention for ... - GitHub



[https://github.com/hako-mikan/sd-webui-traintrain](https://github.com/hako-mikan/sd-webui-traintrain)



[19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969#:~:text=There%27s%20an%20content-safety%20filter%20extension,Was%20this%20translation%20helpful)



Is there content_safety_filter on api? or .. #11969 - GitHub



[https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11969)



[20](https://www.rundiffusion.com/software-licensing#:~:text=Open%20Source%20Licenses%20%C2%B7%20Automatic1111,MIT%20License)



Open Source Licenses - RunDiffusion



[https://www.rundiffusion.com/software-licensing](https://www.rundiffusion.com/software-licensing)



[21](https://www.youtube.com/watch?v=8cVnooYgpDc#:~:text=InvokeAI%203.0%20,fuctions%20and%20optimized%20their%20UI)



InvokeAI 3.0 - NOW with Nodes, Controlnet and Lora Support



[https://www.youtube.com/watch?v=8cVnooYgpDc](https://www.youtube.com/watch?v=8cVnooYgpDc)



[22](https://www.youtube.com/watch?v=1Iz4F7o6hgQ#:~:text=Newly%20Released%20Invoke%20AI%203,Get%20Invoke%20AI%20here)



Newly Released Invoke AI 3.0 Walkthrough - YouTube



[https://www.youtube.com/watch?v=1Iz4F7o6hgQ](https://www.youtube.com/watch?v=1Iz4F7o6hgQ)



[23](https://invoke-ai.github.io/InvokeAI/#:~:text=Invoke%20Invoke%20Training%20has%20moved,can%20find%20more%20by)



Invoke



[https://invoke-ai.github.io/InvokeAI/](https://invoke-ai.github.io/InvokeAI/)



[24](https://github.com/invoke-ai/InvokeAI/issues/6581#:~:text=,fully%20documented%20except%20in%20code)




[enhancement]: Is there a way to make API calls? #6581 - GitHub



[https://github.com/invoke-ai/InvokeAI/issues/6581](https://github.com/invoke-ai/InvokeAI/issues/6581)



[25](https://www.reddit.com/r/invokeai/comments/1edivme/api_texttoimage_example/#:~:text=API%20text,API)



API text-to-image example? : r/invokeai - Reddit



[https://www.reddit.com/r/invokeai/comments/1edivme/api_texttoimage_example/](https://www.reddit.com/r/invokeai/comments/1edivme/api_texttoimage_example/)



[26](https://github.com/invoke-ai/InvokeAI/issues/2205#:~:text=,fork%20of%20InvokeAI%20but)




[enhancement]: Emulate the automatic1111 remote API · Issue #2205



[https://github.com/invoke-ai/InvokeAI/issues/2205](https://github.com/invoke-ai/InvokeAI/issues/2205)


[27](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=Train%20and%20block%20edit%20and,5) [33](https://github.com/shootthesound/comfyUI-Realtime-Lora#:~:text=License) GitHub - shootthesound/comfyUI-Realtime-Lora: Train and block edit and save LoRAs directly inside

ComfyUI for Z-image Turbo, SDXL, Flux, WAN 2.2, SD 1.5


[https://github.com/shootthesound/comfyUI-Realtime-Lora](https://github.com/shootthesound/comfyUI-Realtime-Lora)



[28](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Basically%20you%20pass%20it%20images,part%20of%20my%20own%20workflow) [29](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=EDIT%20,Lora) [30](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=fly%2C%20using%20your%20local%20install,part%20of%20my%20own%20workflow) [31](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=SD%20Scripts%20%28for%20SDXL%29%3A%20https%3A%2F%2Fgithub.com%2Fkohya) [32](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/#:~:text=Image%3A%20r%2FStableDiffusion%20,image%2FWan%2FFlux%20Dev)



Today I made a Realtime Lora Trainer for Z-image/Wan/Flux Dev : r/StableDiffusion



[https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/](https://www.reddit.com/r/StableDiffusion/comments/1peey4o/today_i_made_a_realtime_lora_trainer_for/)



[34](https://modal.com/docs/examples/comfyapp#:~:text=Run%20Flux%20on%20ComfyUI%20as,into%20a%20scalable%20API%20endpoint)



Run Flux on ComfyUI as an API | Modal Docs



[https://modal.com/docs/examples/comfyapp](https://modal.com/docs/examples/comfyapp)



[35](https://comfy.icu/docs/api#:~:text=Run%20ComfyUI%20with%20an%20API,on%20maintaining%20own%20GPU%20infrastructure)



Run ComfyUI with an API - ComfyICU API



[https://comfy.icu/docs/api](https://comfy.icu/docs/api)



[36](https://docs.comfy.org/registry/api-reference/overview#:~:text=API%20Overview%20%C2%B7%20%E2%80%8B,Was%20this%20page%20helpful)



API Overview - ComfyUI



[https://docs.comfy.org/registry/api-reference/overview](https://docs.comfy.org/registry/api-reference/overview)



8


[37](https://github.com/cocktailpeanut/fluxgym#:~:text=) [38](https://github.com/cocktailpeanut/fluxgym#:~:text=Supported%20Models) [39](https://github.com/cocktailpeanut/fluxgym#:~:text=,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638) [40](https://github.com/cocktailpeanut/fluxgym#:~:text=Dead%20simple%20web%20UI%20for,12GB%2F16GB%2F20GB%29%20support) [41](https://github.com/cocktailpeanut/fluxgym#:~:text=,script%20powered%20by%20Kohya%20Scripts) [43](https://github.com/cocktailpeanut/fluxgym#:~:text=1,it%20uses%20Kohya%20Scripts%20underneath) [44](https://github.com/cocktailpeanut/fluxgym#:~:text=%2A%20September%2016%3A%20Added%20,com%2Fcocktailpeanut%2Fstatus%2F1833881392482066638) [45](https://github.com/cocktailpeanut/fluxgym#:~:text=2.%20The%20AI,it%20uses%20Kohya%20Scripts%20underneath) [47](https://github.com/cocktailpeanut/fluxgym#:~:text=https%3A%2F%2Fpinokio) [48](https://github.com/cocktailpeanut/fluxgym#:~:text=News)

GitHub - cocktailpeanut/fluxgym: Dead simple FLUX LoRA training UI
with LOW VRAM support

[https://github.com/cocktailpeanut/fluxgym](https://github.com/cocktailpeanut/fluxgym)



[42](https://www.reddit.com/r/StableDiffusion/comments/1ekb9qj/simpletuner_v098_quantised_flux_training_in_40/#:~:text=SimpleTuner%20v0,potato%20LoRA%20of%20your%20dreams)



SimpleTuner v0.9.8: quantised flux training in 40 gig.. 24 gig.. 16 gig ...



[https://www.reddit.com/r/StableDiffusion/comments/1ekb9qj/simpletuner_v098_quantised_flux_training_in_40/](https://www.reddit.com/r/StableDiffusion/comments/1ekb9qj/simpletuner_v098_quantised_flux_training_in_40/)



[46](https://huggingface.co/XLabs-AI/flux-lora-collection#:~:text=Text,generation%20%20%20%2016)



XLabs-AI/flux-lora-collection · Hugging Face



[https://huggingface.co/XLabs-AI/flux-lora-collection](https://huggingface.co/XLabs-AI/flux-lora-collection)



[49](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod#:~:text=How%20to%20Train%20a%20Flux,UI%20and%20begin%20the)



How to Train a Flux LoRA with FluxGym on RunPod - Next Diffusion



[https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod](https://www.nextdiffusion.ai/tutorials/how-to-train-a-flux-lora-with-fluxgym-on-runpod)


9


