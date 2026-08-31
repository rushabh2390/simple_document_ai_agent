import http from 'k6/http';
import { check, sleep } from 'k6';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';

export function handleSummary(data) {
  return {
    'summary.html': htmlReport(data, { title: 'Local Agent LLM Load Test' }),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
export const options = {
  stages: [
    { duration: '1m', target: 2 },  // Ramp up to 2 concurrent users
    { duration: '3m', target: 4 },   // Stress test with 5 concurrent users
    { duration: '1m', target: 0 },  // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<45000'], // 95% of requests should complete within 5s
  },
};

export default function () {
  const url = 'http://localhost:8000/vault/chat'; // Replace with your FastAPI endpoint
  const payload = JSON.stringify({"messages":[{"role":"user","content":"give me quarterly sales report of year 2004 in table format."}],"retrieval_k":3,"temperature":0.1,"top_k":40});

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '60s',
  };

  const res = http.post(url, payload, params);
  console.log(`Status: ${res.status}, Body: ${res.body}`);
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(1);
}