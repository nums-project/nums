These instructions explain how to setup a Ray cluster with NumS on AWS. 
They supplement the well commented ```autoscaler-aws.yaml``` file.
Refer to the [ray autoscaler](https://docs.ray.io/en/master/cluster/cloud.html) page for additional details. 

Prior to using this autoscaler, nums and boto3 must be installed on your local machine (```pip install nums boto3```) with your AWS credentials configured, as described in the [boto docs](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html). If you use conda, make sure to activate the correct conda environment.

# Steps for successfully launching a NumS cluster on AWS:
To launch a NumS cluster on AWS, follow these steps in specified order.

## A. Configuration
In the ```autoscaler-aws.yaml``` file: 
1. Modify the ```max_workers``` key to set the global max workers that may launch in addition to the head node.
2. Modify the parameters in the ```provider``` field for your aws specific configurations regarding regions and availability zones. 
3. In the ```available_node_types``` field, edit the ```node_config``` field for ```nums.head.default```. 
This will configure the head node. 
You can choose the ec2 instance type, disk and AMI for the head node here. 
Similarly, for the ```nums.worker.default``` field, edit the ```min_workers``` key 
to set the number of NumS workers. 
Edit worker node configurations here under ```node_config``` field. 
Make sure to use the correct AMI as per your region and availability zones. 
4. Modify the ```file_mounts``` field to indicate any directories or files to copy from the local machine to the every node on the cluster, or leave it commented.

## B. Running the configuration file

5. To launch a NumS cluster, run 
```
ray up autoscaler-aws.yaml
```
* This will launch a ray cluster with NumS. 

## C. Example
After you launch the cluster using the steps above, you can refer to [this example](https://github.com/nums-project/nums/blob/main/autoscaler/example.py).
* In this example:
  * We first inititialize ray and set the nums cluster shape.
  * Then we create two nums arrays with random values that get created in a distributed fashion on the worker nodes of the cluser. 
  * We then perform a sum operation on the two nums arrays.
  * Then we do a ```SUM.get()``` which waits and fetches final values of ```X+Y```.

6. To run this example:
  * First run the following commands from your local machine to copy the example.py file to head node of the cluster \
  ```cd <local_path_to_this_repo>/autoscaler``` \
  ```ray rsync-up autoscaler-aws.yaml 'example.py' '/home/ubuntu'```.
  * Then ssh on the head node of the cluster ```ray attach autoscaler-aws.yaml```.
  * Then on the head node, run ```python example.py``` to run this example on the cluster.
  * Then ```exit``` to terminate the ssh connection to the head node. 


## D. Destroying the cluster
7. To destroy the cluster, run the following from your local machine.
```
ray down autoscaler-aws.yaml
```
* Tip: In ```autoscler-aws.yaml``` file, set ```cache_stopped_nodes``` key  to ```False``` in the ```provider``` field to terminate the nodes and disable reuse, instead of shutting them down upon running step 7.
