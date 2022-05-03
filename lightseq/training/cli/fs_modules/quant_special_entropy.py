import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from lightseq.training.ops.pytorch.cross_entropy_layer import LSCrossEntropyLayer
from lightseq.training.ops.pytorch.quantization import TensorQuantizer

@dataclass
class QuantSpecialEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    kl_oneside: bool = field(
        default=False,
        metadata={"help": "if symm_kd"},
    )
    kl_alpha: float = field(
        default=1,
    )

@register_criterion(
    "quant_special_entropy",
    dataclass=QuantSpecialEntropyCriterionConfig,
)
class QuantSpecialEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        kl_oneside=False,
        kl_alpha=1,
        # report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.kl_oneside = kl_oneside
        self.kl_alpha = kl_alpha
        # self.report_accuracy = report_accuracy
        config = LSCrossEntropyLayer.get_config(
            max_batch_tokens=task.args.max_tokens,
            padding_idx=self.padding_idx,
            epsilon=label_smoothing,
            fp16=task.args.fp16,
            local_rank=task.args.device_id,
        )
        self.ls_cross_entropy = LSCrossEntropyLayer(config)

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        def disable_quant(m):
            if isinstance(m, TensorQuantizer):
                m.disable()
                m.disable_quant()
                m.disable_calib()
                m.disable_clip()
                
        def qat_mode(m):
            if isinstance(m, TensorQuantizer):
                m.enable()
                m.enable_quant()
                m.disable_calib()
                m.enable_clip()

        quant_output = model(**sample["net_input"])
        target = model.get_targets(sample, quant_output)
        target = target.to(torch.int32)
        quant_loss, quant_nll_loss = self.ls_cross_entropy(quant_output[0], target)

        model.apply(disable_quant)
        dequant_output = model(**sample["net_input"])
        dequant_loss, dequant_nll_loss = self.ls_cross_entropy(dequant_output[0], target)
        model.apply(qat_mode)
        
        if self.kl_oneside:
            dequant_output[0].detach_()

        symm_kl = self._get_symm_kl(quant_output[0], dequant_output[0])
        loss = 0.5 * (quant_loss + dequant_loss) + self.kl_alpha * symm_kl

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "quant_loss":quant_loss.data,
            "dequant_loss":dequant_loss.data,
            "symm_kl":symm_kl.data,
            "nll_loss": quant_nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
 
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        quant_loss_sum = sum(log.get("quant_loss", 0) for log in logging_outputs)
        dequant_loss_sum = sum(log.get("dequant_loss", 0) for log in logging_outputs)
        symm_kl_sum = sum(log.get("symm_kl", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "quant_loss", quant_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dequant_loss", dequant_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "symm_kl", symm_kl_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
