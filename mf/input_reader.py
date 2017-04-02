#/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import abc

def create_reader(flags, worker_count, worker_index):
    if flags.reader_type == 'raw_text_file':
        return TFTextLineReader(flags.batch_size, flags.train_data, num_epochs=flags.num_epochs)
    elif flags.reader_type == 'kafka':
        return KafkaReader(
                batch_size=flags.batch_size,
                topic=flags.kafka_topic,
                bootstrap_servers=flags.kafka_bootstrap_servers.split(','),
                group_id=flags.kafka_group_id,
                consumer_count=worker_count,
                consumer_index=worker_index,
                max_partition_fetch_bytes=flags.kafka_max_partition_fetch_bytes,
                auto_offset_reset=flags.kafka_auto_offset_reset,
                auto_metric = flags.auto_metric,
                metric_interval = flags.metric_interval,
                seek_to_end = flags.seek_to_end)
    else:
        return None

class InputReader(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, batch_size):
        self._batch_size = batch_size

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def read_batch(self, output_list, sess):
        pass

    @abc.abstractmethod
    def is_read_finished(self):
        pass

class KafkaReader(InputReader):
    def __init__(self,
            batch_size,
            topic,
            bootstrap_servers,
            group_id,
            consumer_count=1,
            consumer_index=0,
            max_partition_fetch_bytes=1048576,
            auto_offset_reset = 'earliest',
            timeout=None,
            auto_metric = False,
            metric_interval = 60,
            seek_to_end=False):
        super(KafkaReader, self).__init__(batch_size)
        self._batch_size = batch_size
        self._topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._consumer_count = consumer_count
        self._consumer_index = consumer_index
        self._max_partition_fetch_bytes = max_partition_fetch_bytes
        self._auto_offset_reset = auto_offset_reset
        self._timeout = timeout
        self._auto_metric = auto_metric
        self._metric_interval = metric_interval
        self._seek_to_end = seek_to_end

    def init(self):
        print 'Bootstrap server is {}'. format(self._auto_offset_reset)
        from tf_util.kafka import kafka_reader
        self._reader = kafka_reader.KafkaReader(
            topic=self._topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            consumer_num =self._consumer_count,
            auto_offset_reset=self._auto_offset_reset,
            consumer_index =self._consumer_index,
            max_partition_fetch_bytes=self._max_partition_fetch_bytes,
            auto_metric = self._auto_offset_reset,
            metric_interval = self._metric_interval,
            seek_to_end = self._seek_to_end)
        return True

    def read_batch(self, output_list, sess):
        receive = self._reader.get_messages(self._batch_size, self._timeout)
        length = len(receive)
        for i in range(length):
            output_list[i] = receive[i][1]
        return length

    def is_read_finished(self):
        return False

class TFTextLineReader(InputReader):
    def __init__(self, batch_size, input_path, num_epochs=None,
            num_threads=2, capacity=10000):
        super(TFTextLineReader, self).__init__(batch_size)
        self._input_path = input_path
        self._reader = None
        self._read_finished = False
        self._num_threads = num_threads
        self._capacity = capacity
        self._num_epochs = num_epochs

    def init(self):
        import tensorflow as tf
        list_file_op = tf.train.match_filenames_once(self._input_path)
        filename_queue = tf.train.string_input_producer(list_file_op, shared_name='input_file_name_queue', num_epochs=self._num_epochs)
        reader = tf.TextLineReader()
        file_name, read_line_op = reader.read(filename_queue)
        self._read_batch_line_op = tf.train.batch(tensors=[read_line_op],
                batch_size=self._batch_size,
                capacity=self._capacity,
                num_threads=self._num_threads,
                allow_smaller_final_batch=True)
        return True

    def read_batch(self, output_list, sess):
        import tensorflow as tf
        try:
            lines = sess.run(self._read_batch_line_op)
            for i, line in enumerate(lines):
                output_list[i] = line
            return i
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            self._read_finished = True
            return 0

    def is_read_finished(self):
        return self._read_finished


class RawTextFileReader(InputReader):
    def __init__(self, batch_size, text_file_path):
        super(RawTextFileReader, self).__init__(batch_size)
        self._text_file_path = text_file_path
        self._reader = None
        self._read_finished = False

    def init(self):
        self._reader = open(self._text_file_path, 'r')
        return True

    def read_batch(self, output_list):
        for i in range(0, self._batch_size):
            line = self._reader.readline()
            if not line:
                print 'Input file read finished.'
                self._read_finished = True
                return i
            output_list[i] = line
        return self._batch_size

    def is_read_finished(self):
        return self._read_finished
