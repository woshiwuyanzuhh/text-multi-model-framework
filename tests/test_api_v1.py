"""
test_api - API 接口测试

测试用例 - test_* - 单元测试（测试最小执行单元 --- 函数）

conda install pytest -c conda-forge

Author: Grant Johnny
Version: 0.0.1
"""
import os

import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"
API_KEY = os.getenv('API_KEY', '')
AUTH_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}


@pytest.mark.smoke
def test_base_url():
    """测试健康检查端点"""
    resp = requests.get(BASE_URL + '/health')
    assert resp.status_code == 200


@pytest.mark.api
def test_predict_valid():
    """测试 /predict 接口"""
    resp = requests.post(
        url=BASE_URL + '/api/v1/predict',
        headers=AUTH_HEADERS,
        json={
            'text': '湖北省黄冈市09届高三年级期末考试试题',
        },
    )
    assert resp.status_code == 200
    result = resp.json()  # type: dict
    assert 'label' in result
    assert result.get('code') == 0 and result.get('message') == 'OK'


@pytest.mark.api
def test_predict_invalid():
    """测试 /predict 接口"""
    resp = requests.post(
        url=BASE_URL + '/api/v1/predict',
        headers=AUTH_HEADERS,
        json={},
    )
    assert resp.status_code == 200
    result = resp.json()  # type: dict
    assert 'label' not in result
    assert result.get('code') == -20 and result.get('message') != 'OK'


@pytest.mark.api
def test_predict_text_too_long():
    """测试输入文本超过最大长度限制（应返回 code -11）"""
    long_text = '测试' * 3000  # 6000 字，超过默认 MAX_TEXT_LENGTH=5000
    resp = requests.post(
        url=BASE_URL + '/api/v1/predict',
        headers=AUTH_HEADERS,
        json={'text': long_text},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -11
    assert 'label' not in result


@pytest.mark.api
def test_predict_no_auth():
    """测试无认证头（应返回 code -1，HTTP 401）"""
    resp = requests.post(
        url=BASE_URL + '/api/v1/predict',
        headers={'Content-Type': 'application/json'},
        json={'text': '测试文本'},
    )
    assert resp.status_code == 401
    result = resp.json()
    assert result.get('code') == -1
