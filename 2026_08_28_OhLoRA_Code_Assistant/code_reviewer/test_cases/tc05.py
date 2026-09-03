
# 1. exception ignored

try:
    # 오류가 날 수 있는 코드
    num = int(input("숫자 입력: "))
except:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    num = int(input("숫자 입력: "))
except Exception:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    num = int(input("숫자 입력: "))
except BaseException as exp:
    # 예외를 삼켜버리는 경우 (에러를 무시함)
    pass

try:
    # 오류가 날 수 있는 코드
    num = int(input("숫자 입력: "))
except:
    # exception occured 출력하는 경우
    print('exception occurred')

try:
    num = int(input("숫자 입력: "))
except IOError:
    pass

try:
    num = int(input("숫자 입력: "))
except IOError as ioe:
    pass
