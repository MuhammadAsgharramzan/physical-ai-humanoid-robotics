---
name: "AI-Native Docusaurus Textbook Chapter Creation"
description: "A reusable skill for writing high-quality textbook chapters in Docusaurus, integrating RAG chatbot, personalization, and Urdu translation."
version: "1.0.0"
---

## When to Use This Skill
- Writing chapters/modules for AI-native textbooks
- Preparing content compatible with Docusaurus
- Embedding RAG chatbot prompts and personalization
- Creating content with code examples and practical exercises
- Developing curriculum for technical subjects like robotics and AI

## Process Steps
1. Outline chapter key points, theory, labs, exercises, and code snippets.
2. Write each section clearly, explaining concepts first, then examples.
3. Add diagrams or descriptive placeholders for visuals.
4. Embed RAG chatbot prompts where user questions are expected.
5. Add personalization hooks at chapter start.
6. Prepare Urdu translations for all text content.
7. Format the markdown for Docusaurus compliance (navigation, headings, links).
8. Review and ensure modularity and consistency.

## Output Format
- Docusaurus-compatible markdown file for chapter
- Code snippets in fenced blocks
- Lab instructions and exercises as markdown lists
- RAG chatbot prompts embedded inline

## Example
**Input:** Outline for ROS 2 fundamentals chapter
**Output:** Complete Docusaurus markdown with theory, exercises, code snippets, and chatbot hooks

## Detailed Implementation Guidelines

### 1. Chapter Structure
- Start with frontmatter containing id, title, sidebar_label, and description
- Include clear learning objectives at the beginning
- Organize content in logical sections with appropriate headings
- End with summary and exercises

### 2. Theoretical Content
- Provide comprehensive explanations of core concepts
- Use analogies and real-world examples to clarify complex topics
- Include historical context where relevant to show evolution of concepts
- Connect theory to practical applications

### 3. Practical Examples and Code Snippets
- Include working code examples that students can implement
- Provide step-by-step implementation guides
- Add comments and explanations for complex code sections
- Show expected outputs and common troubleshooting tips

### 4. Hands-on Labs and Exercises
- Design practical exercises that reinforce theoretical concepts
- Include difficulty levels and estimated completion times
- Provide clear instructions and expected outcomes
- Add extension activities for advanced students

### 5. RAG Chatbot Integration
- Identify points in the text where students might have questions
- Include inline prompts that would be helpful for a RAG system
- Structure content with clear headings and subheadings for retrieval
- Add FAQ sections addressing common student questions

### 6. Personalization Hooks
- Include optional advanced topics for different skill levels
- Add pathways for different learning styles (visual, hands-on, theoretical)
- Provide choices in exercises to accommodate different interests
- Include self-assessment checkpoints

### 7. Urdu Translation Preparation
- Structure content to be easily translatable
- Avoid idioms and complex cultural references that don't translate well
- Include technical terms with English explanations
- Plan for right-to-left text formatting where necessary

### 8. Docusaurus Formatting
- Use proper markdown formatting for headings, lists, and code blocks
- Include appropriate metadata in frontmatter
- Use Docusaurus-specific features like tabs, admonitions, and callouts
- Ensure proper linking between related content

## Quality Assurance Checklist
- [ ] Learning objectives clearly stated
- [ ] Content appropriate for target audience
- [ ] Code examples are functional and well-commented
- [ ] Exercises have clear instructions and solutions
- [ ] Content is modular and can stand alone
- [ ] Proper Docusaurus formatting applied
- [ ] RAG chatbot prompts embedded appropriately
- [ ] Personalization options included
- [ ] Translation considerations addressed

## Common Pitfalls to Avoid
- Avoid overly complex jargon without proper explanation
- Don't include code without context or explanation
- Avoid assuming prerequisite knowledge without verification
- Don't create exercises without clear success criteria
- Avoid monolithic content that can't be reused
- Don't forget to test code examples in real environments