// RAG Agent Workflow
digraph {
	analyze [label="Initial Analysis"]
	sql [label="SQL Query Agent"]
	category [label="Category Agent"]
	final [label="Final Response"]
	analyze -> sql
	sql -> category
	category -> final
}
