# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from MulticlassClassifier import MulticlassClassifier


@register_criterion('label_smoothed_cross_entropy_with_classifier')
class LabelSmoothedCrossEntropyCriterionWithClassifier(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, classifier_lambda, class_name, clf_model_path):
        super().__init__(task, sentence_avg, label_smoothing)
        self.classifier_lambda = classifier_lambda
        self.class_name = class_name
        self.clf = MulticlassClassifier(device="cpu", model_path=clf_model_path)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--class-name', default='fairytale', type=float, metavar='D', help='class for the classifier loss')
        parser.add_argument('--clf-model-path', default='models/multiclass', type=float, metavar='D', help='path to clf model')
        parser.add_argument('--classifier-lambda', default=1., type=float, metavar='D', help='weight for the classifier loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        classifier_loss = self.compute_classifier_loss(sample, net_output)

        logging_output['classifier_loss'] = utils.item(classifier_loss.data)
        loss += self.classifier_lambda * classifier_loss

        return loss, sample_size, logging_output

    def compute_classifier_loss(self, sample, net_output):
        loss = -self.clf.predict(net_output, [self.class_name])
        
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        classifier_loss_sum = utils.item(sum(log.get('classifier_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('classifier_loss', classifier_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
