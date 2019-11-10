# Flask API for Neural Style Transfer and Pose Estimation 

`Style Transfer`와 `Pose Estimation`을 **Flask Restful API**로 구성해 보았습니다.

`Pytorch` 기반으로, GPU 기반입니다. 

`Style Transfer`의 경우 Pytorch의 [튜토리얼 문서](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)를 기반으로 클래스 형식으로 구현하였고,

`Pose Estimation`의 경우 [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)라는 오픈소스 프로젝트를 url 기반으로 데이터를 받을 수 있도록 수정하여 사용하였습니다. 

## 사용법 