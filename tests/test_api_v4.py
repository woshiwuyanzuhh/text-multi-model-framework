"""
test_api_v4 - V4 大模型接口测试

测试用例覆盖：
- 正常预测（需 LLM 可用）
- 空文本
- 无效 JSON
- 认证失败（无 Authorization 头 / 错误 Key）
- 超时场景（需服务端配置短超时）

运行方式：
    API_KEY="your-key" pytest tests/test_api_v4.py -v

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

# 是否有可用的 LLM API Key（决定 V4 接口是否可正常调用）
LLM_AVAILABLE = bool(os.getenv('OPENAI_API_KEY'))


@pytest.mark.smoke
def test_base_url():
    """测试健康检查端点"""
    resp = requests.get(BASE_URL + '/health')
    assert resp.status_code == 200


@pytest.mark.api
def test_predict_valid():
    """测试 V4 正常预测（需 LLM 可用，否则跳过）"""
    if not LLM_AVAILABLE:
        pytest.skip('OPENAI_API_KEY 未设置，跳过 V4 正常预测测试')
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={'text': '央行宣布降低存款准备金率 0.5 个百分点'},
        timeout=60,
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == 0
    assert result.get('message') == 'OK'
    assert 'label' in result


@pytest.mark.api
def test_predict_empty_text():
    """测试 V4 空文本（应返回 code -10）"""
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={'text': ''},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -10
    assert 'label' not in result


@pytest.mark.api
def test_predict_invalid_json():
    """测试 V4 无效 JSON 请求体（应返回 code -20）"""
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -20
    assert 'label' not in result


@pytest.mark.api
def test_predict_text_too_long():
    """测试 V4 输入文本超过最大长度限制（应返回 code -11）"""
    long_text = '测试' * 3000  # 6000 字，超过默认 MAX_TEXT_LENGTH=5000
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={'text': long_text},
        timeout=30,
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -11
    assert 'label' not in result


@pytest.mark.api
def test_predict_no_auth():
    """测试 V4 无认证头（应返回 code -1，HTTP 401）"""
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers={'Content-Type': 'application/json'},
        json={'text': '测试文本'},
    )
    assert resp.status_code == 401
    result = resp.json()
    assert result.get('code') == -1


@pytest.mark.api
def test_predict_wrong_auth():
    """测试 V4 错误 API Key（应返回 code -1，HTTP 401）"""
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer wrong-key-xxx',
        },
        json={'text': '测试文本'},
    )
    assert resp.status_code == 401
    result = resp.json()
    assert result.get('code') == -1


@pytest.mark.api
def test_predict_llm_unavailable():
    """测试 V4 LLM 不可用（OPENAI_API_KEY 未设置时应返回 code -43）

    若测试环境已设置 OPENAI_API_KEY，则跳过此用例。
    """
    if LLM_AVAILABLE:
        pytest.skip('OPENAI_API_KEY 已设置，V4 LLM 可用，跳过不可用场景测试')
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={'text': '测试文本'},
        timeout=30,
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -43


@pytest.mark.api
@pytest.mark.skipif(
    not os.getenv('TMF_TEST_TIMEOUT'),
    reason='超时测试需服务端配置 LLM_TIMEOUT=1 且指向不可达地址，设置 TMF_TEST_TIMEOUT=1 启用',
)
def test_predict_timeout():
    """测试 V4 超时场景（需服务端配置 LLM_TIMEOUT=1 且 LLM 端点不可达）

    前置条件：
    1. 服务端设置 LLM_TIMEOUT=1
    2. 服务端设置 LLM_BASE_URL 指向不可达地址（如 http://10.255.255.1:8080/v1）
    3. 设置环境变量 TMF_TEST_TIMEOUT=1 启用此测试

    预期结果：code -40，响应时间 < 15s
    """
    resp = requests.post(
        url=BASE_URL + '/api/v4/predict',
        headers=AUTH_HEADERS,
        json={'text': '测试文本'},
        timeout=30,
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result.get('code') == -40
