# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed ImageNet training and validation, with model replicas.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from inception import inception_distributed_train
from inception.imagenet_data import ImagenetData

flags = tf.app.flags

flags.DEFINE_string("data_dir", "/imagenet",
                    "Directory for storing mnist data")
flags.DEFINE_integer("worker_index", 0,
                     "Worker task index, should be >= 0. worker_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_string("ps_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")

FLAGS = flags.FLAGS

def main(unused_argv):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.worker_index)

  if FLAGS.job_name == "ps":
    server.join()
    sys.exit(0)
    
  # `worker` jobs will actually do the work.
  dataset = ImagenetData(subset=FLAGS.subset)
  assert dataset.data_files()
  # Only the chief checks for or creates train_dir.
  if FLAGS.task_id == 0:
    if not tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.MakeDirs(FLAGS.train_dir)
      inception_distributed_train.train(server.target, dataset, cluster_spec)

  num_workers = len(worker_hosts)
  worker_grpc_url = 'grpc://' + worker_hosts[0]
  print("Worker GRPC URL: %s" % worker_grpc_url)
  print("Worker index = %d" % FLAGS.worker_index)
  print("Number of workers = %d" % num_workers)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
