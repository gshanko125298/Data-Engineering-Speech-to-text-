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
