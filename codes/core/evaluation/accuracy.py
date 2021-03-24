import numpy as np


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def confusion_matrix(y_pred, y_real):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    label_map = {label: i for i, label in enumerate(label_set)}
    confusion_mat = np.zeros((num_labels, num_labels), dtype=np.int64)
    for rlabel, plabel in zip(y_real, y_pred):
        index_real = label_map[rlabel]
        index_pred = label_map[plabel]
        confusion_mat[index_real][index_pred] += 1

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_acc(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)), len(lb_set)


def top_k_hit(score, lb_set, k=3):
    idx = np.argsort(score)[-k:]
    return len(lb_set.intersection(idx)) > 0, 1


def top_k_accuracy(scores, labels, k=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    for kk in k:
        hits = []
        for x, y in zip(scores, labels):
            y = [y] if isinstance(y, int) else y
            hits.append(top_k_hit(x, set(y), k=kk)[0])
        res.append(np.mean(hits))
    return res


def get_weighted_score(score_list, coeff_list):
    """Get weighted score with given scores and coefficients.
    Given n predictions by different classifier: [score_1, score_2, ...,
    score_n] (score_list) and their coefficients: [coeff_1, coeff_2, ...,
    coeff_n] (coeff_list), return weighted score: weighted_score =
    score_1 * coeff_1 + score_2 * coeff_2 + ... + score_n * coeff_n
    Args:
        score_list (list[list[np.ndarray]]): List of list of scores, with shape
            n(number of predictions) X num_samples X num_classes
        coeff_list (list[float]): List of coefficients, with shape n.
    Return:
        list[np.ndarray]: List of weighted scores.
    """
    assert len(score_list) == len(coeff_list)
    num_samples = len(score_list[0])
    for i in range(1, len(score_list)):
        assert len(score_list[i]) == num_samples

    scores = np.array(score_list)  # (num_coeff, num_samples, num_classes)
    coeff = np.array(coeff_list)  # (num_coeff, )
    weighted_scores = list(np.dot(scores.T, coeff).T)
    return weighted_scores
