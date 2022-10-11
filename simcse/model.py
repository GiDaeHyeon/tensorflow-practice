"""SimCSE Model"""
import tensorflow as tf
from transformers import TFBertPreTrainedModel, TFBertModel


class SimCSEModel(TFBertPreTrainedModel):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config.model
        self.bert = TFBertModel.from_pretrained(self.config.weight)

    def call(self, inputs):
        return self.bert.call()

    def reduce_representation(self, representations):
        hidden_size = tf.shape(representations)[-1]

        ctx = tf.distribute.get_replica_context()
        if ctx and ctx.num_replicas_in_sync != 1:
            print(f"reduce reprsentations, num_replicas_in_sync: {ctx.num_replicas_in_sync}, and id: {ctx.replica_id_in_sync_group}")
            representations = tf.where(
                (tf.range(0, ctx.num_replicas_in_sync) == ctx.replica_id_in_sync_group)[:, tf.newaxis, tf.newaxis],
                tf.expand_dims(representations, 0),
                tf.expand_dims(tf.zeros_like(representations), 0),
            )
            [representations] = ctx.all_reduce(tf.distribute.ReduceOp.SUM, [representations])
            representations = tf.reshape(representations, [1, -1, hidden_size])
        else:
            representations = tf.expand_dims(representations, 0)

        return representations
