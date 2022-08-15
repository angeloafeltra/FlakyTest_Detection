import subprocess
import gc
from ml_utils import collect_available_choices

def get_configurations():
    to_run = []
    choices = collect_available_choices()
    #these are flakeFlagger projects
    choices["data"] = ["Achilles", "Activiti", "alluxio", "ambari", "elastic-job-lite", "hbase", "hector", "http-request",
                       "httpcore", "incubator-dubbo", "Java-WebSocket", "logback", "ninja", "okhttp", "orbit", "spring-boot",
                       "undertow", "wro4j", "flakeFlagger"]
    
    #these are iDFlakies project
    #choices["data"] = ["Activiti", "admiral", "aletheia", "elastic-job-lite", "fastjson", "hadoop", "http-request", "incubator-dubbo",
        #"Java-WebSocket", "pippo", "querydsl", "Struts", "wildfly", "idFlakies"]
        
#these are the project for run the same pipeline in RQ4
    #choices["data"] = ['Alluxio-alluxio','spring-projects-spring-boot','elasticjob-elastic-job-lite','apache-incubator-dubbo','square-okhttp','activiti-activiti',
     #                  'ninjaframework-ninja','doanduyhai-Achilles','undertow-io-undertow','kevinsawicki-http-request','qos-ch-logback',
      #                 'orbit-orbit','apache-ambari','hector-client-hector','wro4j-wro4j','apache-hbase','tootallnate-java-websocket','apache-httpcore','flakeFlagger']

    with open('configuration.txt', 'r') as f:
        for line in f.readlines():
            params = {}
            conf = line.strip().split(" ")
            # default
            for key in choices.keys():
                params[key] = "none"
            params["k"] = 10
            params["feature_sel"] = "none"
            for c in conf:
                for key in choices.keys():
                    if c in choices[key]:
                        params[key] = c
            to_run.append(params)
    return to_run
    
    
# run
for conf in get_configurations():
    subprocess.run(["python", "./ml_main.py", "-i", conf["data"], "-k", str(conf["k"]), "-p", conf["feature_sel"], conf["balancing"], conf["optimization"], conf["classifier"]])
    gc.collect()
                    

