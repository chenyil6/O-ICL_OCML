import nltk
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import json
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

# 初始化词形还原器
# lemmatizer = WordNetLemmatizer()

result_file_name = "sqqr-tags-n-v-index.json"


# 分词、词形还原、POS标注函数
def preprocess(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    return pos_tags


# 提取名词、动词、介词（排除be动词和助动词）
def extract_words(pos_tags, pos_type):
    words = [word for word, pos in pos_tags if
             pos.startswith(pos_type) and word not in ['be', 'is', 'are', 'do', 'does']]
    return words


def extract_pos(tokens):
    nouns = []
    adjective = []
    verbs = []
    interrogative = []
    # pos_tags = nltk.pos_tag(tokens)
    for word, pos in tokens:
        if pos.startswith('N') and word not in ['Is', 'Are']:
            nouns.append(word)
        elif pos.startswith('J'):
            adjective.append(word)
        elif pos.startswith('V') and word not in ['be', 'is', 'are', 'do', 'does']:
            verbs.append(word)
        elif pos.startswith('W') or word in ['Is', 'Are']:
            interrogative.append(word)
    # result = interrogative + nouns + verbs + adjective
    result = nouns + verbs
    return result


# 加载预训练的Word2Vec模型
word2vec_path = "/data/pjw/model/W2V/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def create_vector(question_tokens):
    vectors = [word2vec_model[word] for word in question_tokens if word in word2vec_model]
    if vectors:
        # 计算向量的平均值作为问题向量
        question_vector = sum(vectors) / len(vectors)
        return question_vector.tolist()
    else:
        # 如果问题中的所有单词都不在Word2Vec词汇中，返回空向量
        return [0.0] * 300  # 使用300维的零向量表示


# 加载vqav2数据集
with open("/data/pyz/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json", 'r') as train_file:
    train_data = json.load(train_file)
with open("/data/share/pyz/data/vqav2/v2_mscoco_val2014_question_subdata.json", 'r') as test_file:
    test_data = json.load(test_file)

# 预处理和分词训练集和测试集问题
train_vectors = []
for train_item in tqdm(train_data['questions'], desc="Processing Train Data"):
    train_question = train_item['question']
    processed_train_question = preprocess(train_question)
    train_vector = extract_pos(processed_train_question)
    train_vector = create_vector(train_vector)
    train_vectors.append(train_vector)

test_vectors = []
for test_item in tqdm(test_data['questions'], desc="Processing Test Data"):
    test_question = test_item['question']
    processed_test_question = preprocess(test_question)
    test_vector = extract_pos(processed_test_question)
    test_vector = create_vector(test_vector)
    test_vectors.append(test_vector)

# 创建Annoy索引
embedding_dim = 300  # Word2Vec向量维度
annoy_index = AnnoyIndex(embedding_dim, 'angular')  # 使用angular距离度量

# 将训练集的向量添加到Annoy索引中
for i in range(len(train_data['questions'])):
    train_vector = np.array(train_vectors[i])  # 转换为NumPy数组
    annoy_index.add_item(i, train_vector)

# 构建Annoy索引
annoy_index.build(n_trees=10)  # n_trees越大，索引越准确但速度越慢

# 创建一个字典来存储相似问题
similar_questions_dict = {}

# 遍历测试集数据并添加进度条显示
for i, test_item in tqdm(enumerate(test_data['questions']), total=len(test_data['questions']),
                         desc="Processing Similarity"):
    test_question_id = test_item['question_id']  # 提取测试集问题ID
    test_vector = np.array(test_vectors[i])

    # 进行近似搜索
    similar_indices = annoy_index.get_nns_by_vector(test_vector, 32, search_k=-1)

    # 提取训练集问题的对应ID并存储到字典中
    # similar_question_ids = [train_data['questions'][index]['question_id'] for index in similar_indices]
    similar_question_ids = [index for index in similar_indices]
    similar_questions_dict[test_question_id] = similar_question_ids

# 将字典写入JSON文件
with open(result_file_name, 'w') as json_file:
    json.dump(similar_questions_dict, json_file, indent=4)

print("Similar question IDs for vqav2 dataset have been written to " + result_file_name)

# # 计算两个句子的相似度
# def calculate_similarity(text1, text2):
#     vectorizer = CountVectorizer().fit_transform([text1, text2])
#     vectors = vectorizer.toarray()
#     similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
#     return similarity[0][0]

# def calculate_cosine_similarity(tokens1, tokens2):
#     vectorizer = CountVectorizer(tokenizer=lambda text: text, lowercase=False)
#     vectors = vectorizer.fit_transform([tokens1, tokens2])
#     cosine_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
#     return cosine_sim

# train_question_ids = []

# # 遍历训练集数据
# for j, train_item in tqdm(enumerate(train_data['questions'])):
#     train_question_id = train_item['question_id']  # 提取训练集问题ID
#     train_nouns = train_nouns_list[j]
#     train_verbs = train_verbs_list[j]
#     train_prepositions = train_prepositions_list[j]

#     # noun_similarity = calculate_similarity(' '.join(test_nouns), ' '.join(train_nouns))
#     # verb_similarity = calculate_similarity(' '.join(test_verbs), ' '.join(train_verbs))
#     # preposition_similarity = calculate_similarity(' '.join(test_prepositions), ' '.join(train_prepositions))
#     # total_similarity = noun_similarity + verb_similarity + preposition_similarity

#     cosine_sim = calculate_cosine_similarity(test_nouns + test_verbs + test_prepositions,
#                                              train_nouns + train_verbs + train_prepositions)
# similarities.append(cosine_sim)
# train_question_ids.append(train_question_id)

# top_similar_indices = np.argsort(similarities)[-32:]  # 选取相似度最高的32个训练集问题的索引

# # 提取训练集问题的对应ID并存储到字典中
# top_similar_question_ids = [train_question_ids[i] for i in top_similar_indices]
# similar_questions_dict[test_question_id] = top_similar_question_ids

# # 将字典写入JSON文件
# with open('similar_question_ids_vqav2.json', 'a') as json_file:
#     json.dump(similar_questions_dict, json_file, indent=4)
#     json_file.write('\n')  # 写入换行，以便每个结果占据一行

