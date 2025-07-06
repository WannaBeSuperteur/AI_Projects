
import pandas as pd

if __name__ == '__main__':
    test_csv = pd.read_csv('test_final.csv')
    rag_data = list(set(list(test_csv['rag_retrieved_data'])))  # 중복 제거
    rag_data_str = '\n'.join(rag_data)

    with open('../memory/rag_data_text.txt', 'w', encoding='UTF8') as f:
        f.write(rag_data_str)
        f.close()
