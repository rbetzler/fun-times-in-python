#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import docker

def run_python_in_docker(docker_container, airflow_script):

    client = docker.from_env()

    client.start(docker_container)

    py_client = client.exec_create(container = docker_container, cmd = 'python ' + airflow_script)

    client.exec_start(py_client['Id'])

    client.stop(docker_container)

run_python_in_docker(sys.argv[1], sys.argv[2])
