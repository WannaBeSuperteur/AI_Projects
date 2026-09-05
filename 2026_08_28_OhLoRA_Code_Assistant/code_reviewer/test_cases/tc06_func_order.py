
# [안티패턴 예시]
def process_user_data(user_id):
    # 최초 작성된 기본 사용자 처리
    pass


def order_consumer_ranking():  # line 28 함수와 유사
    pass


def send_welcome_email(user_id):
    pass


# 6개월 후 추가
def process_user_data_v2(user_id, options):
    # 기존 함수와 유사하지만 비즈니스 로직 변경 버전
    pass


# 1년 후 추가
def process_user_data_batch(user_ids):
    # 대량 처리를 위해 추가된 함수
    pass


def rank_customer():  # line 8 함수와 유사
    pass


if __name__ == '__main__':
    process_user_data('111')
    send_welcome_email('123')
