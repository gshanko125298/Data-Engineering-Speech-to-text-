# Data-Engineering-Speech-to-text-
To recognizing the value of large data sets for speech-t0-text data sets, and seeing the opportunity that there are many text corpuses for Amharic and Swahili languages, and understanding that complex data engineering skills is valuable to your profile for employers, this week’s task is simple: design and build a robust, large scale, fault tolerant, highly available Kafka cluster that can be used to post a sentence and receive an audio file.  By the end of this project, you should produce a tool that can be deployed to process posting and receiving text and audio files from and into a data lake, apply transformation in a distributed manner, and load it into a warehouse in a suitable format to train a speech-t0-text model.  
Key Features

# Speech processing

            Automatic Speech Recognition (ASR)
                    Supported models: Jasper, QuartzNet, CitriNet, Conformer-CTC, Conformer-Transducer, Squeezeformer-CTC, Squeezeformer-Transducer, # ContextNet, LSTM-Transducer (RNNT), LSTM-CTC, ...
                    Supports CTC and Transducer/RNNT losses/decoders
                    Beam Search decoding
                    Language Modelling for ASR: N-gram LM in fusion with Beam Search decoding, Neural Rescoring with Transformer
                    Streaming and Buffered ASR (CTC/Transducer) - Chunked Inference Examples

            Speech Classification and Speech Command Recognition: MatchboxNet (Command Recognition)
            Voice activity Detection (VAD): MarbleNet
            Speaker Recognition: TitaNet, ECAPA_TDNN, SpeakerNet

            Speaker Diarization
                    Clustering Diarizer: TitaNet, ECAPA_TDNN, SpeakerNet
                    Neural Diarizer: MSDD (Multi-scale Diarization Decoder)

            Pretrained models on different languages.: English, Spanish, German, Russian, Chinese, French, Italian, Polish, ...
            NGC collection of pre-trained speech processing models.

# Natural Language Processing
            NeMo Megatron pre-training of Large Language Models
            Neural Machine Translation (NMT)
            Punctuation and Capitalization
            Token classification (named entity recognition)
            Text classification
            Joint Intent and Slot Classification
            Question answering
            GLUE benchmark
            Information retrieval
            Entity Linking
            Dialogue State Tracking
            Prompt Learning
            NGC collection of pre-trained NLP models.

# Speech synthesis (TTS)
            Spectrogram generation: Tacotron2, GlowTTS, TalkNet, FastPitch, FastSpeech2, Mixer-TTS, Mixer-TTS-X
            Vocoders: WaveGlow, SqueezeWave, UniGlow, MelGAN, HiFiGAN, UnivNet
            End-to-end speech generation: FastPitch_HifiGan_E2E, FastSpeech2_HifiGan_E2E
            NGC collection of pre-trained TTS models.

 # Tools
            Text Processing (text normalization and inverse text normalization)
            CTC-Segmentation tool
            Speech Data Explorer: a dash-based tool for interactive exploration of ASR/TTS datasets
            
            Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.
# Requirements

    Python 3.8 or above
    Pytorch 1.10.0 or above
    NVIDIA GPU for training
Tutorials

A great way to start with NeMo is by checking one of our tutorials.
Getting help with NeMo

FAQ can be found on NeMo's Discussions board. You are welcome to ask questions or start discussions there.
Installation
Conda

We recommend installing NeMo in a fresh Conda environment.

conda create --name nemo python==3.8
conda activate nemo

# Install PyTorch using their configurator.

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

Note

The command used to install PyTorch may depend on your system.
Pip

Use this installation mode if you want the latest released version.

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']

Note

Depending on the shell used, you may need to use "nemo_toolkit[all]" instead in the above command.
Pip from source

Use this installation mode if you want the a version from particular GitHub branch (e.g main).

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]

From source

Use this installation mode if you are contributing to NeMo.

apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/NVIDIA/NeMo
cd NeMo
./reinstall.sh

Note

If you only want the toolkit without additional conda-based dependencies, you may replace reinstall.sh with pip install -e . when your PWD is the root of the NeMo repository.
RNNT

Note that RNNT requires numba to be installed from conda.

conda remove numba
pip uninstall numba
conda install -c conda-forge numba

Megatron GPT

Megatron GPT training requires NVIDIA Apex to be installed.

git clone https://github.com/ericharper/apex.git
cd apex
git checkout nm_v1.11.0
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

# NeMo Text Processing

NeMo Text Processing, specifically (Inverse) Text Normalization, requires Pynini to be installed.

bash NeMo/nemo_text_processing/install_pynini.sh

# Docker containers:

To build a nemo container with Dockerfile from a branch, please run

# DOCKER_BUILDKIT=1 docker build -f Dockerfile -t nemo:latest .

If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 22.08-py3 and then installing from GitHub.

docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:22.08-py3

Examples

Many examples can be found under "Examples" folder.
Contributing

We welcome community contributions! Please refer to the CONTRIBUTING.md CONTRIBUTING.md for the process.
Publications

We provide an ever growing list of publications that utilize the NeMo framework. Please refer to PUBLICATIONS.md. We welcome the addition of your own articles to this list !
Citation

@article{kuchaiev2019nemo,
  title={Nemo: a toolkit for building ai applications using neural modules},
  author={Kuchaiev, Oleksii and Li, Jason and Nguyen, Huyen and Hrinchuk, Oleksii and Leary, Ryan and Ginsburg, Boris and Kriman, Samuel and Beliaev, Stanislav and Lavrukhin, Vitaly and Cook, Jack and others},
  journal={arXiv preprint arXiv:1909.09577},
  year={2019}
}

License

NeMo is under Apache 2.0 license.
