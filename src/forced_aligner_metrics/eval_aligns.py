import warnings
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Iterator, Dict, Any, List
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from collections import Counter
from scipy.stats import entropy



class W2V2Embedding:
    
    def __init__(self, model_str, layer_range = (12, 16), device = "cuda"):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_str)
        self.model = Wav2Vec2Model.from_pretrained(model_str).to(self.device)
        self.sampling_rate = self.processor.sampling_rate
        self.layer_range = layer_range
        self.model.eval()
        # Freeze parameters so the forward pass builds no autograd graph.
        # With frozen params and non-grad inputs this is equivalent to
        # running under no_grad, without importing a specific framework.
        self.model.requires_grad_(False)

    def _feat_extract_output_lengths(self, input_lengths):
        """Compute exact wav2vec2 frame counts in numpy.

        Mirrors ``Wav2Vec2Model._get_feat_extract_output_lengths`` using the
        convolutional feature encoder's kernel sizes and strides, avoiding a
        direct dependency on the underlying tensor framework.
        """
        lengths = np.asarray(input_lengths)
        for kernel_size, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            lengths = (lengths - kernel_size) // stride + 1
        return lengths

    def extract_embeddings_batch(self, audios):
        results = []

        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)

        lo, hi = self.layer_range
        hidden = np.stack(
            [h.cpu().numpy() for h in outputs.hidden_states[lo:hi]]
        ).mean(0)

        # sample lengths
        input_lengths = attention_mask.sum(-1).cpu().numpy()

        # exact wav2vec2 frame lengths
        frame_lengths = self._feat_extract_output_lengths(input_lengths)

        for i in range(len(audios)):

            num_samples = int(input_lengths[i])
            duration = num_samples / self.sampling_rate
            num_frames = int(frame_lengths[i])

            emb = hidden[i, :num_frames]

            # frame timestamps
            frame_times = np.linspace(0,duration,num_frames,endpoint=False)

            results.append((emb, frame_times))

        return results
    
    

def normalize_intervals(intervals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Tuple[float, float, str]]]:
    normalized = {}
    for tier_name, tier_intervals in (intervals or {}).items():
        normalized_tier = []
        for entry in tier_intervals:
            if isinstance(entry, dict):
                start = float(entry.get("start", entry.get("start_time", 0.0)))
                end = float(entry.get("end", entry.get("end_time", start)))
                label = str(entry.get("text", "")).strip()
            else:
                continue

            if not label:
                label = "sil"
            normalized_tier.append((start, end, label))
        normalized[tier_name] = normalized_tier
    return normalized


def assign_labels_to_frames(frame_times, intervals):
    labels = []
    j = 0
    for t in frame_times:
        while j < len(intervals) - 1 and t > intervals[j][1]:
            j += 1
        labels.append(intervals[j][2])
    return labels


def pool_emb(x, k=2):
    n=len(x)//k
    if n==0: return x
    return x[:n*k].reshape(n,k,-1).mean(1)

def crop_frames(emb, frame_times, start, end):
    idx=(frame_times>=start)&(frame_times<=end)
    return emb[idx]

def dtw_sim(x,y):
    if len(x)<2 or len(y)<2: return None
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-8)
    x,y=pool_emb(x),pool_emb(y)
    distance, path =fastdtw(x,y,dist=cosine)
    return 1 - distance/len(path)

@dataclass
class ForcedAlignEvalConfig:
    """Configuration for the import-based evaluation API.

    Parameters
    ----------
    model:
        Hugging Face model name used for Wav2Vec2 embeddings.
    device:
        Device string passed to the underlying model (e.g. ``"cpu"`` or
        ``"cuda"``). The choice of compute backend is left to the
        ``transformers`` library and its installed framework.
    layer_range:
        Hidden layer range to average when extracting embeddings.
    batch_size:
        Number of samples processed per embedding batch.
    samples_phon_eval:
        Number of phone-level evaluation samples to collect.
    max_frames:
        Maximum frame count used for clustering-based PCMI evaluation.
    n_clusters:
        Number of clusters used by the PCMI clustering step.
    samples_word_eval:
        Number of word-level samples to collect for WACS.
    max_words:
        Maximum number of vocabulary items used for WACS.
    min_occ:
        Minimum occurrence count required for a word to enter the WACS vocabulary.
    max_pairs:
        Maximum number of positive/negative DTW pairs per word.
    """

    model : str = "facebook/mms-300m"
    device : str = "cuda"
    layer_range: Tuple[int, int] = (15, 16)
    batch_size : int = 2

    samples_phon_eval : int = 50
    max_frames : int = 10_000
    n_clusters : int = 50

    samples_word_eval : int = 200
    max_words: int = 200
    min_occ : int = 3
    max_pairs: int = 10
    

class ForcedAlignEval:
    """Main import-based evaluator for PCMI and WACS metrics.

    Parameters
    ----------
    config:
        A ForcedAlignEvalConfig instance defining the evaluation behavior.
    """

    def __init__(self, config=ForcedAlignEvalConfig()):
        self.config=config
        self.emb_model = W2V2Embedding(config.model, config.layer_range, config.device)
    
    def collect_embeddings(self, audios: Iterator[np.ndarray], intervals_iter: Iterator[Dict[str, List[Dict[str, Any]]]]) -> dict:

        all_frame_embs = []
        all_frame_labels = []
        word2embs = defaultdict(list)

        processed = 0
        sample_size = max(
            self.config.samples_word_eval,
            self.config.samples_phon_eval,
        )
        batch_size = self.config.batch_size

        while processed < sample_size:
            batch_audios = []
            batch_intervals = []

            for _ in range(min(batch_size, sample_size - processed)):
                try:
                    audio = next(audios)
                    intervals = next(intervals_iter)
                except StopIteration:
                    break

                batch_audios.append(audio)
                batch_intervals.append(intervals)

            if not batch_audios:
                break

            batch_results = self.emb_model.extract_embeddings_batch(batch_audios)

            for batch_idx, (emb, frame_times) in enumerate(batch_results):
                normalized_intervals = normalize_intervals(batch_intervals[batch_idx])

                if processed < self.config.samples_phon_eval and "phones" in normalized_intervals:
                    labels = assign_labels_to_frames(frame_times, normalized_intervals["phones"])
                    all_frame_embs.append(emb)
                    all_frame_labels.extend(labels)

                if processed < self.config.samples_word_eval and "words" in normalized_intervals:
                    for start, end, label in normalized_intervals["words"]:
                        label = label.strip().lower()
                        if not label or label in ["sp", "sil", "<unk>"]:
                            continue

                        wemb = crop_frames(emb, frame_times, start, end)
                        if len(wemb) >= 4:
                            word2embs[label].append(wemb)

            processed += len(batch_results)

        return {
            "frame_embeddings": all_frame_embs,
            "frame_labels": all_frame_labels,
            "word_embeddings": word2embs,
        }
        
        
    def compute_metrics(self, audios:Iterator[np.ndarray], intervals_iter: Iterator[Dict[str, List[Dict[str, Any]]]]) -> dict:
        """Compute evaluation metrics from audio arrays and interval dictionaries.

        Parameters
        ----------
        audios:
            Iterator yielding audio arrays.
        intervals_iter:
            Iterator yielding interval dictionaries of the form
            {"phones": [{"text": ..., "start_time": ..., "end_time": ...}],
             "words": [{"text": ..., "start_time": ..., "end_time": ...}]}

        Returns
        -------
        dict
            A dictionary containing `pcmi_score` when phone-level labels are present
            and `wacs_score` when word-level embeddings are available.
        """

        # -------------------------
        # NMI
        # -------------------------
        data = self.collect_embeddings(audios=audios, intervals_iter=intervals_iter)
        all_embeddings=data["frame_embeddings"]
        all_labels=data["frame_labels"]

        nmi=None

        if len(all_embeddings):

            X=np.concatenate(all_embeddings,axis=0)
            y=np.array(all_labels)

            if len(X)>self.config.max_frames:

                label_to_indices=defaultdict(list)

                for i,label in enumerate(y):
                    label_to_indices[label].append(i)

                freqs={
                    label:len(indices)
                    for label,indices in label_to_indices.items()
                }

                weights={
                    label:np.sqrt(freq)
                    for label,freq in freqs.items()
                }

                total_weight=sum(weights.values())
                selected=[]

                for label,indices in label_to_indices.items():

                    n=int(self.config.max_frames*weights[label]/total_weight)
                    n=min(n,len(indices))

                    chosen=np.random.choice(indices, n, replace=False)

                    selected.extend(chosen)

                selected=np.array(selected)

                X=X[selected]
                y=y[selected]

            norms=np.linalg.norm(X,axis=1,keepdims=True)
            norms[norms==0]=1

            X=(X/norms).astype(np.float16)

            cluster_ids=MiniBatchKMeans(
                n_clusters=self.config.n_clusters,
                batch_size=1000
            ).fit_predict(X)

            nmi= {"pcmi_score": float(normalized_mutual_info_score(y, cluster_ids))}

            label_counts=Counter(y)
            label_probs=np.array(list(label_counts.values()),dtype=np.float64)
            label_probs/=label_probs.sum()

            label_ent=float(entropy(label_probs))
            label_ent_norm=float(label_ent/np.log(len(label_probs)))

            cluster_purity=[]

            for k in range(self.config.n_clusters):

                idx=(cluster_ids==k)

                if idx.sum()==0:
                    continue

                labels_k=y[idx]
                counts=Counter(labels_k)

                purity=max(counts.values())/len(labels_k)
                cluster_purity.append(purity)

            nmi.update({
                "pcmi_normalized_label_entropy":label_ent_norm,
                "pcmi_num_frames":int(len(X)),
                "pcmi_num_labels":int(len(label_counts)),
            })
        # -------------------------
        # WACS
        # -------------------------
        word2embs=data["word_embeddings"]

        pos_sims=[]
        neg_sims=[]

        vocab=[
            w for w,v in word2embs.items()
            if len(v)>=self.config.min_occ
        ]


        vocab=sorted(
            vocab,
            key=lambda x:len(word2embs[x]),
            reverse=True
        )[:self.config.max_words]

        for word in vocab:

            embs=word2embs[word]
            pairs=0

            for i in range(len(embs)):
                for j in range(i+1,len(embs)):

                    sim=dtw_sim(embs[i],embs[j])

                    if sim is not None:
                        pos_sims.append(sim)

                    pairs+=1

                    if pairs>=self.config.max_pairs:
                        break

                if pairs>=self.config.max_pairs:
                    break

            neg_words=np.random.choice(
                vocab,
                min(5,len(vocab)),
                replace=False
            )

            for nw in neg_words:

                if nw==word:
                    continue

                x=embs[np.random.randint(len(embs))]
                y=word2embs[nw][
                    np.random.randint(len(word2embs[nw]))
                ]

                sim=dtw_sim(x,y)

                if sim is not None:
                    neg_sims.append(sim)

        wacs=None

        if len(pos_sims) and len(neg_sims):

            pos=float(np.mean(pos_sims))
            neg=float(np.mean(neg_sims))

            # ---- WACS enrichments ----
            pos_std=float(np.std(pos_sims))
            neg_std=float(np.std(neg_sims))

            # Cohen's d
            eps=1e-8
            pooled=np.sqrt(
                (
                    ((len(pos_sims)-1)*(pos_std**2))+
                    ((len(neg_sims)-1)*(neg_std**2))
                )/
                max(1,(len(pos_sims)+len(neg_sims)-2))
            )+eps

            cohen_d=(pos-neg)/pooled

            # overlap estimate
            wacs_margin=pos-neg

            wacs={
                "wacs_score":wacs_margin,
                "wacs_positive_dtw":pos,
                "wacs_negative_dtw":neg,
                "wacs_num_positive_pairs":int(len(pos_sims)),
                "wacs_num_negative_pairs":int(len(neg_sims)),
                "wacs_cohen_d":float(cohen_d),
                "wacs_vocab_size": len(vocab)
            }
                        
        if nmi and wacs:
            nmi.update(wacs)
        elif wacs:
            nmi = wacs
        if not nmi:
            warnings.warn("No tiers phones or words", UserWarning)
            
        return nmi