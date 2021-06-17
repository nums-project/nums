This readme contains the instructions on how you can setup a NumS cluster on AWS. They supplement the well commented ```autoscaler-aws.yaml``` file.
Refer to the [ray autoscaler](https://docs.ray.io/en/master/cluster/cloud.html) page for additional details. 

Note, prior to using this autoscaler, nums and boto3 must be installed on your local machine ```pip install nums boto3``` with your AWS credentials configured, as described in the [boto docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html). If you use conda, make sure to activate the correct conda environment.

## Configuration

* Modify the ```max_workers``` key to enforce the global max workers that could launch in addition to the head node, or leave it commented.

* Modify the keys in ```provider``` field for your aws specific configurations regarding regions and availability zones. 

* In the ```available_node_types``` field, edit the ```node_config``` field for ```nums.head.default```. This will configure the head node. You can choose the ec2 instance type, disk and AMI for head node here. Similarly, for the ```nums.worker.default``` field, edit the ```min_workers``` key to set the number of NumS workers. Edit the node configurations here as well under ```node_config``` field. (Make sure to use the correct AMI as per your region and availability zones).

* Then modify the ```file_mounts``` field to indicate any directories or files to copy from the local machine to the every node on the cluster. Leave commented otherwise.

## Running

To launch a NumS cluster, run 
```
ray up autoscaler-aws.yaml
```
This will launch a ray cluster with NumS. 

And to destroy the cluster, run
```
ray down autoscaler-aws.yaml
```
* set ```cache_stopped_nodes``` to ```False``` in the ```provider``` field to terminate the nodes instead of a shut-down and disable reuse.

To ssh into the head node of the cluster after it is set up, run
```
ray attach autoscaler-aws.yaml
```

## Example
After you set up the cluster using the steps above, you can refer to [this example](https://github.com/nums-project/nums/blob/main/autoscaler/example.py).
* In this example
  * We first inititialize ray and set the nums cluster shape.
  * Then we create two nums arrays with random values that get created in a distributed fashion on the worker nodes of the cluser. 
  * We then perform a sum operation on the two nums arrays.
  * Then we do a ```SUM.get()``` which waits and fetches final values of ```X+Y```.

* To run this example:
  * First run the following command from your local machine to copy the example.py file to head node of the cluster \
  ```ray rsync_up autoscaler_aws.yaml '/home/ubuntu/example.py' '<local_path_to_this_repo>/autoscaler/example.py'```.
  * Then ssh on the head node of the cluster ```ray attach autoscaler-aws.yaml```.
  * Then on the head node, run ```python nums/autoscaler/example.py``` to run this example on the cluster.
