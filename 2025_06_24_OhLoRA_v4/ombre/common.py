
from stylegan.run_stylegan_vectorfind_v7 import get_property_change_vectors as get_property_change_vectors_v7
from stylegan.run_stylegan_vectorfind_v8 import get_property_change_vectors as get_property_change_vectors_v8


# Property Score Change Vector 의 값을 찾아서 반환
# Create Date : 2025.06.26
# Last Update Date : -

# Arguments:
# - vectorfind_ver (str) : StyleGAN-VectorFind 버전 ('v7' or 'v8')

# Returns:
# - eyes_vectors  (dict(NumPy Array)) : eyes (눈을 뜬 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - mouth_vectors (dict(NumPy Array)) : mouth (입을 벌린 정도) 속성값을 변화시키는 벡터 정보 (각 그룹 별)
# - pose_vectors  (dict(NumPy Array)) : pose (고개 돌림) 속성값을 변화시키는 벡터 정보 (각 그룹 별)

def get_property_score_change_vectors(vectorfind_ver):
    if vectorfind_ver == 'v7':
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v7()
    else:  # v8
        eyes_vectors, mouth_vectors, pose_vectors = get_property_change_vectors_v8(vectorfind_ver)

    return eyes_vectors, mouth_vectors, pose_vectors
