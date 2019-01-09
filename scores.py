# @author Runlong Yu, Mingyue Cheng, Weibo Gao

import numpy as np
import math

def count(tup_list, tup_index):
    count = 0
    for x in range(tup_index, len(tup_list)):
        if tup_list[x][1] == tup_list[tup_index][1]:
            count += 1
        else:
            break
    return count

def topK_scores(test, predict, topk, user_count, item_count):
    PrecisionSum = np.zeros(topk+1)
    RecallSum = np.zeros(topk+1)
    F1Sum = np.zeros(topk+1)
    NDCGSum = np.zeros(topk+1)
    OneCallSum = np.zeros(topk+1)
    DCGbest = np.zeros(topk+1)
    MRRSum = 0
    MAPSum = 0
    total_test_data_count = 0

    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    for i in range(user_count):
        user_test = []
        user_predict = []
        test_data_size = 0
        for j in range(item_count):
            if test[i * item_count + j] == 1.0:
                test_data_size += 1
            user_test.append(test[i * item_count + j])
            user_predict.append(predict[i * item_count + j])
        if test_data_size == 0:
            continue
        else:
            total_test_data_count += 1

        temp_dic = {}
        x = 1
        for temp in user_predict:
            temp_dic[x] = temp
            x += 1
        temp_dic = sorted(temp_dic.items(), key = lambda x: x[1], reverse = True)
        predict_max_num_index_list = []
        total_count = 0
        tup_index = 0
        while total_count < topk:
            index_count = count(temp_dic, tup_index)
            if index_count > topk - total_count:
                temp_lists = np.arange(index_count)
                np.random.shuffle(temp_lists)
                for temp_list in range(topk - total_count):
                    predict_max_num_index_list.append(temp_dic[tup_index + temp_lists[temp_list]][0] - 1)
                break
            else:
                for sum_num in range(0, index_count):
                    predict_max_num_index_list.append(temp_dic[tup_index + index_count - 1][0] - 1)
                total_count += index_count
                tup_index = total_count
        hit_sum = 0
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)

        for k in range(1, topk + 1):
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1] #
            if user_test[item_id] == 1:
                hit_sum += 1
                DCG[k] += 1 / math.log(k + 1)
            # precision, recall, F1, 1-call
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            f1 = 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            F1Sum[k] += float(f1)
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]
            if hit_sum > 0:
                OneCallSum[k] += 1
            else:
                OneCallSum[k] += 0
        # MRR
        p = 1
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                break
            p += 1
        MRRSum += 1 / float(p)
        # MAP
        p = 1
        AP = 0.0
        hit_before = 0
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] == 1:
                AP += 1 / float(p) * (hit_before + 1)
                hit_before += 1
            p += 1
        MAPSum += AP / test_data_size
    print('MAP:', MAPSum / total_test_data_count)
    print('MRR:', MRRSum / total_test_data_count)
    print('Prec@%d:' % topk, PrecisionSum[topk] / total_test_data_count)
    print('Rec@%d:' % topk, RecallSum[topk] / total_test_data_count)
    print('F1@%d:' % topk, F1Sum[topk] / total_test_data_count)
    print('NDCG@%d:' % topk, NDCGSum[topk] / total_test_data_count)
    print('1-call@%d:' % topk, OneCallSum[topk] / total_test_data_count)
    return
