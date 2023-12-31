
Running the SS xApp
=======================

.. warning::

    If you already have SS xApp deployed on your system, you need to restart the pod using the command below before running the rest of the commands. If you are freshly deploying the xApp, you can skip this step.

.. code-block:: bash

    sudo kubectl -n ricxapp rollout restart deployment ricxapp-secure-slicing

Terminal 1: Start the Core Network/Add Ues to Network Namespace

.. code-block:: bash

    sudo ip netns add ue1
    sudo ip netns add ue2
    sudo ip netns add ue3
    sudo ip netns add ue4
    sudo ip netns list    
    sudo srsepc 

Terminal 2: Set up Environment Variables and Base Station

.. code-block:: bash

    export E2NODE_IP=`hostname  -I | cut -f1 -d' '`
    export E2NODE_PORT=5006
    export E2TERM_IP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath='{.items[0].spec.clusterIP}'`
    
.. code-block:: bash
       
    sudo srsenb \
    --enb.n_prb=50 --enb.name=enb1 --enb.enb_id=0x19B --rf.device_name=zmq \
    --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2000,rx_port=tcp://localhost:2009,id=enb,base_srate=23.04e6" \
    --ric.agent.remote_ipv4_addr=${E2TERM_IP} --log.all_level=warn --ric.agent.log_level=debug --log.filename=stdout \
    --ric.agent.local_ipv4_addr=${E2NODE_IP} --ric.agent.local_port=${E2NODE_PORT} \
    --slicer.enable=1 --slicer.workshare=0

Terminal 3: Set up the first UE

.. code-block:: bash

    sudo srsue --rf.device_name=zmq --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2010,rx_port=tcp://localhost:2006,id=ue1,base_srate=23.04e6" --gw.netns=ue1 --usim.algo=xor --usim.imsi=001010123456789

Terminal 4: Set up the second UE

.. code-block:: bash

    sudo srsue --rf.device_name=zmq --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2011,rx_port=tcp://localhost:2007,id=ue2,base_srate=23.04e6" --gw.netns=ue2 --usim.algo=xor --usim.imsi=001010123456780

Terminal 5: Set up the third UE

.. code-block:: bash

    sudo srsue --rf.device_name=zmq --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2012,rx_port=tcp://localhost:2008,id=ue3,base_srate=23.04e6" --gw.netns=ue3 --usim.algo=xor --usim.imsi=001010123456781

Terminal 5: Setup the fourth UE

.. code-block:: bash

	 sudo srsue --rf.device_name=zmq --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2013,rx_port=tcp://localhost:2005,id=ue4,base_srate=23.04e6" --gw.netns=ue4 --usim.algo=xor --usim.imsi=001010123456782

Terminal 5: Start the gnuradio flowgraph

.. code-block:: bash

    python3 4UE.py

Terminal 6 & 7: Set up iperf3 test on the server side

.. code-block:: bash
   
   iperf3 -s -B 172.16.0.1 -p 5006 -i 1
   iperf3 -s -B 172.16.0.1 -p 5020 -i 1
   iperf3 -s -B 172.16.0.1 -p 5030 -i 1
   iperf3 -s -B 172.16.0.1 -p 5040 -i 1

Terminal 8 & 9: Set up iperf3 test on the client side

We add an additional bandwidth argument "-b xxM" on each iperf3 test on client side to create a scenario of UEs trying to access more or less of resources on the network. If a UE surpasses the pre-determined threshold for amount of data packets transmitted, it is considered as Malicious by the ZTRAN xApp.

.. code-block:: bash

//run the script instead
//depends on the situation you want either RTT or throughput

   sudo ip netns exec ue1 iperf3 -c 172.16.0.1 -p 5006 -i 1 -t 36000 -R -b 30M
   sudo ip netns exec ue2 iperf3 -c 172.16.0.1 -p 5020 -i 1 -t 36000 -R -b 10M
   sudo ip netns exec ue3 iperf3 -c 172.16.0.1 -p 5030 -i 1 -t 36000 -R -b 10M
   sudo ip netns exec ue4 iperf3 -c 172.16.0.1 -p 5040 -i 1 -t 36000 -R -b 10M

You should notice traffic flow on both the server and client side for both UEs. Move on to the next step.

Terminal 10

.. code-block:: bash
    
    cd secure-slicing
    export KONG_PROXY=`sudo kubectl get svc -n ricplt -l app.kubernetes.io/name=kong -o jsonpath='{.items[0].spec.clusterIP}'`
    export E2MGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2mgr-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export APPMGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-appmgr-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export E2TERM_SCTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath='{.items[0].spec.clusterIP}'`
    export ONBOARDER_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-xapp-onboarder-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export RTMGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-rtmgr-http -o jsonpath='{.items[0].spec.clusterIP}'`

Deploying the xApp
------------------

.. code-block:: bash

    curl -L -X POST "http://$KONG_PROXY:32080/onboard/api/v1/onboard/download" --header 'Content-Type: application/json' --data-binary "@nexran-onboard.url"
    curl -L -X GET "http://$KONG_PROXY:32080/onboard/api/v1/charts"
    curl -L -X POST "http://$KONG_PROXY:32080/appmgr/ric/v1/xapps" --header 'Content-Type: application/json' --data-raw '{"xappName": "ztran"}'

Add another terminal to print the logs

.. code-block:: bash

    sudo kubectl logs -f -n ricxapp -l app=ricxapp-secure-slicing

.. warning::
    Before running the rest of the commands, detach two of the terminals with the iperf3 test running for 2 UEs to observe the downlink traffic.
    Also, detach the terminal with the logs.

Now run the test script with the following commands. You have to access the test script through the root directory to execute the commands in the script. The test script has commands for creating NodeB, UEs, and slices within the xApp, as well as binding the UEs to the slices. The xApp runs it's authentication mechanism for identifying authorized UEs during the creation of UEs.

.. code-block:: bash

    chmod +x /ss-scripts/zmqfourue.sh
    ./ss-scripts/zmqfourue.sh

After a short time you can observe through the logs that UE1 will be considered malicious and moved to a different slice. You also observe the traffic exchange for UE1 will significantly decrease. 

To observe the throughput changes graphically, save the iperf3 test results for both UEs on text files and run the follwing python script. Make sure to change the file name on the script to match your files.

.. code-block:: bash
    
    python3 iperfplot.py
		
An example of graph generated from running ZTRAN for with 2 UEs (one regular and one malicious) is given below:
 
 .. image:: ORAN-secure-slicing.png
    :width: 80%
    :alt: OAIC Secure Slicing Xapp


