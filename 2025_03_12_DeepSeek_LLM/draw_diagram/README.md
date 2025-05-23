Diagram 생성 알고리즘

## 실행 방법

```commandline
python test_draw_diagram.py
```

* 입력
  * ```diagram.txt``` - 다이어그램 포맷으로 작성된 텍스트 파일
  * 아래 내용은 LLM 이 다양한 포맷으로 답변을 작성할 때 이를 최대한 커버하기 위한 테스트 케이스임

```python
model = [
    [1, 100, 100, "circle", 100, 80, "dashed line", (255, 0, 0), (0, 0, 0), [2, 3]],
    [2, 100, 500, "round rectangle", 100, 80, dashed line, ('0', '255', '0'), ('0', '0', '0'), [3, 4]],
    [3, 400, 100, rectangle, 120, 100, "solid arrow", ("0", "0", "255"), ("0", "0", "0"), [1, 2]],
    [4, 400, 500, rectangle, 120, 100, "solid line", (0, 255, 255), (0, 0, 0), [2, 3]],
    [5, 650, 100, "round rectangle", 120, 100, "solid arrow", (255, 255, 0), (0, 0, 0), ["4", "5"]],
    ["6", '650', '500', "circle", "120", '100', "solid arrow", "(255, 0, 255)", "(0, 0, 0)", '[5, 6]'],
    [7, 900, 300, "rectangle", 140, 140, "solid arrow", (0, 255, 255), (0, 0, 0), "[6, 7]"],
```

* 출력
  * ```diagram.png``` - 생성된 다이어그램 이미지