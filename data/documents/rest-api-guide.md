# REST API Best Practices

## 1. Endpoint Design

### Resource-Oriented Design
REST APIs should be designed around resources, not actions.

- Resources: /users, /posts, /comments
- Use nouns, not verbs
- Avoid /get_user (wrong) instead use GET /users/{id} (correct)
- Hierarchical relationships: /users/{id}/posts

### HTTP Methods
- GET: Retrieve a resource (safe, idempotent)
- POST: Create a new resource
- PUT: Replace an entire resource (idempotent)
- PATCH: Partial update to a resource
- DELETE: Remove a resource (idempotent)
- HEAD: Same as GET but without response body
- OPTIONS: Describe communication options

### Status Codes
- 200 OK: Successful request
- 201 Created: Resource created successfully
- 204 No Content: Successful but no content to return
- 400 Bad Request: Invalid request format
- 401 Unauthorized: Authentication required
- 403 Forbidden: Authenticated but no permission
- 404 Not Found: Resource doesn't exist
- 500 Internal Server Error: Server error
- 503 Service Unavailable: Server temporarily down

## 2. Authentication and Authorization

### Authentication Methods
- Basic Auth: Username and password in headers (use HTTPS only)
- Bearer Tokens: JWT or OAuth tokens
- API Keys: Simple but less secure
- OAuth 2.0: Industry standard for delegated access
- OpenID Connect: Authentication layer on top of OAuth 2.0

### Security Best Practices
- Always use HTTPS
- Implement rate limiting
- Use short-lived tokens with refresh tokens
- Validate all inputs
- Implement CORS appropriately
- Use secure headers (X-Frame-Options, Content-Security-Policy)
- Regular security audits

## 3. Data Format and Versioning

### Error Responses
Consistent error format aids client implementation:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request",
    "details": [
      {"field": "email", "message": "Invalid email format"}
    ]
  }
}
```

### API Versioning
- URL path versioning: /api/v1/users
- Header versioning: Accept: application/vnd.api+json;version=1
- Query parameter: /users?api_version=1

## 4. Documentation

### OpenAPI/Swagger
- Standardized API documentation
- Interactive testing capability
- Code generation support
- API discovery features

### Documentation Best Practices
- Clear endpoint descriptions
- Example requests and responses
- Authentication requirements
- Rate limits and throttling
- Error handling guidelines
- Pagination strategies

## 5. Performance Optimization

### Pagination
- Limit default results (e.g., 20 items)
- Support offset/limit or cursor-based pagination
- Include total count for offset-based
- Use cursor pagination for large datasets

### Filtering and Sorting
- Support query parameters for filtering
- Allow multiple sort fields
- Document filter capabilities
- Implement efficient database queries

### Caching
- Use ETag and Last-Modified headers
- Implement cache-control strategies
- Support HTTP caching
- Consider CDN integration

## 6. Rate Limiting

### Implementation
- X-RateLimit-Limit: Total requests allowed
- X-RateLimit-Remaining: Requests remaining
- X-RateLimit-Reset: Time when limit resets
- 429 Too Many Requests: When limit exceeded

### Strategies
- Per-user rate limiting
- Per-IP rate limiting
- Sliding window algorithm
- Token bucket algorithm
