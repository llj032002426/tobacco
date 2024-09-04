import cloudconvert


def svg2emf(file_name):
    # 1.定义cloudconvert的api_key(token)
    api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiZWMwNTVhM2Y3OTkwNmJkYTNjMWVmNjRlYzk1OGY5Y2JiMDc1ZDI5ZjFkOGY2YTk3ODNlNjBiMjZhY2U2NzY0Mjc4NDY2M2E2YWY2NmE2YzEiLCJpYXQiOjE3MjEzNzk0MzMuMTU0MDI0LCJuYmYiOjE3MjEzNzk0MzMuMTU0MDI1LCJleHAiOjQ4NzcwNTMwMzMuMTUwNzQ5LCJzdWIiOiI2OTAzOTQwMCIsInNjb3BlcyI6WyJ1c2VyLnJlYWQiLCJ1c2VyLndyaXRlIiwidGFzay5yZWFkIiwidGFzay53cml0ZSIsIndlYmhvb2sucmVhZCIsIndlYmhvb2sud3JpdGUiLCJwcmVzZXQucmVhZCIsInByZXNldC53cml0ZSJdfQ.iAXHy6BH6wO51HqEau1f2KIMOUhjhDxOqbRiwts6CO2VeCPtLa-KsvocHovsdAojmzDsW_q7m75sczLTPG3E5QUeFC8vfw2COZQU370PdihMcOKAoRJebLZaHDsuYOAmmhi4a1NCBb72q18FUCIGQE8YKefXubvCc-4e1RnNyhG4WWq8yuWtEU4Jmjix4WxK1lxCWQP9uqnsXpOuWp5QdVrbO0EDB2IYmonr_44RdrfVlb51xEdbVbfK1iQ4hBgKke6FC0eLV-CQqEsVx2tsuTVpQjAb_-uOBB5eUFGDjdE0rgaWDRBneR4bmh4Y_29hLgUg30tR36pw94-htW7xzeS1LJH0MISaWcg5CgM2bw1Xs-sru67mVDvQ8qiseussxDvUVtT6I93BFCRGXCtCllihy3ftNl8Ut-f4WLpgdZYRo_laSnTHTGLBzUnguL7iDDZ8iaaZjGuhEv_wUf-yxkq9r-0BXSUQLPm1DWKmN5mt13vNooSF_nUMM37P6oMX49BMVQUcOtU0idW3z2Sl_FpKV6ftc3WBmslzWXHoVOsIb2TgieoYrfWCNjiTXI3u95SEDvPCZbBMXW8RQKF620CaXU16ZbNv_bTXbU68xZISucf28JQIVkdNjikIMuA-Dr69DTjX8sIzbbDxzsttBoE0DZQ-HNDfgz-dld7jNes'
    # api_key为申请账号的token

    # 2.创建api客户端
    cloudconvert.configure(api_key=api_key, sandbox=False)

    # 3.创建一个job
    job = cloudconvert.Job.create(payload={
        "tasks": {
            # 导入文件
            "import-1": {
                "operation": "import/upload"
            },

            # 转换文件 svg转emf
            "convert-1": {
                "operation": "convert",
                "input_format": "svg",
                "output_format": "emf",
                "engine": "inkscape",
                "input": [
                    "import-1"
                ]
            },

            # 输出文件(将转换的emf文件输出为url)
            "export-1": {
                "operation": "export/url",
                "input": [
                    "convert-1"
                ],
                "inline": False,
                "archive_multiple_files": False
            }
        },
    })

    # 4.启动job对应的上传任务（上传原文件）
    upload_task_id = job['tasks'][0]['id']  # job['tasks'][0]对应的是上传任务，job['tasks'][1]是转换任务
    upload_task = cloudconvert.Task.find(id=upload_task_id)
    cloudconvert.Task.upload(file_name=file_name, task=upload_task)

    # 5.启动job对应的下载任务（下载转换后的文件）
    exported_url_task_id = job['tasks'][2]['id']  # job['tasks'][2]对应的是下载任务
    res = cloudconvert.Task.wait(id=exported_url_task_id)  # 使用cloudconvert.Task的内置函数等待转换文件完成
    file = res.get("result").get("files")[0]  # 使用cloudconvert.Task的内置函数提取转换文件
    cloudconvert.download(filename=resource.replace('svg', 'emf'), url=file['url'])  # 下载文件

    # 6.为了防止云端job数目过多而产生的api请求错误，在job完成后即将其删除
    cloudconvert.Job.delete(job['id'])


if __name__ == '__main__':
    resource = r'./emf_pic/temp_loss2.svg'
    svg2emf(resource)