
# 1. 동일한 인수가 반복되는 코드

# 주문 정보를 처리하는 개별 함수들
def calculate_total_price(item_name, price, quantity, address):
    # 총액 계산 (수량별 할인 등 적용 가능)
    base_price = price * quantity
    print(f"[{item_name}] {quantity}개 기본 금액: {base_price}원")
    return base_price

def calculate_shipping_fee(item_name, price, quantity, address):
    # 배송지에 따른 배송비 계산
    if "제주" in address:
        shipping_fee = 5000
    else:
        shipping_fee = 3000
    print(f"[{address}] 배송비: {shipping_fee}원")
    return shipping_fee

def print_order_summary(item_name, price, quantity, address):
    # 주문 요약 정보 출력
    total = calculate_total_price(item_name, price, quantity, address)
    shipping = calculate_shipping_fee(item_name, price, quantity, address)
    print(f"--- 주문서 ---")
    print(f"상품명: {item_name} / 최종 결제 금액: {total + shipping}원")

# 함수 호출 (매번 4개의 인수를 똑같이 넘겨야 함)
item_name = "맥북 에어"
price = 1200000
quantity = 1
address = "제주도 제주시"

print_order_summary(item_name, price, quantity, address)
