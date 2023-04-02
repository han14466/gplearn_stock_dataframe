#!/usr/bin/env python  
#-*- coding:utf-8 _*-  

import ray


def init_ray(module_list=None):
    self_path = "/".join(__file__.split("/")[:-2])
    if module_list is None:
        module_list = []

    if self_path not in module_list:
        module_list.append(self_path)

    excludes = ["**/*dump.pkl", "**/*.docx", "**/*.csv"]
    runtime_env = {"py_modules": module_list, "excludes": excludes}
    address = "ray://127.0.0.1:10001"
    if not ray.is_initialized():
        ray.init(address=address, runtime_env=runtime_env)
