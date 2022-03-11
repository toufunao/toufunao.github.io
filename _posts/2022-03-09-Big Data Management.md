---
layout:     post   				    # 使用的布局（不需要改）
title:      Big Data Management Review 				# 标题 
subtitle:    #副标题
date:       2022-03-09 				# 时间
author:     Chris 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Data Science
    - Data Management
---

## Lecture1. Database Review 

### What is a good Index?

Access type supported by the index efficiently (equity query, range query)
Access time (query response time)
Insertion time (data record insertion time)
Deletion time (data record deletion time)
Space overhead (size of index file)

### Classification of Indexes

**Primary index**: sequentially ordered file, the index whose search key specifies the sequential order of the file (clustered index)
**Secondary index**: index whose search key specifies an order different from the sequential order of the file (unclustered index)

**Dense Index**: index record appears for every search-key value in the file
**Sparse Index**: contains index records for only some search-key values

### B+ Tree Index

**Basic properties**: Disk-based tree structure, multi-way tree, balanced tree, 

**Advantages**: automatically reorganize itself with small local changes; Reorganization of entire file is not required to maintain performance.
**Disadvantags**: extra insertion and deletion overhead, space overhead

### Hash Index

An ideal hash function is **uniform** and **random**

Bucket overflow: 1. Insufficient buckets     2. skew in distribution of records

It can be handled by **overflow buckets**. Overflow chaining / closed hashing – the overflow buckets of a given bucket are chained together in a linked list.

## Lecture2. Spatial Data Management

### Basic Knowledge

**Spatial Data**: spatial objects may be have spatial extent or they can be points.

Two ways to represent objects with extend, vector representation & raster representation.

**Spatial Relationships**: topological relationships, distance relationships, directional relationships.

| Classification            | Content                                                      |
| ------------------------- | ------------------------------------------------------------ |
| topological relationships | disjoint, intersects, equlas, inside, contains, adjacent     |
| distance relationships    | explicit (Euclidean distance )  or abstract distance class   |
| directional relationships | N, S, E,W etc. left, right, above, below, front, behind etc. |

**Spatial Queries**: Nearest neighbor query, spatial join, range query

**Special about Spatial**: dimensionality , complex spatial extent, no standard definitions of spatial operations and algebra 

**Two-step spatial query processing**: 

Evaluating spatial relationships on geometric data is slow; a *spatial object* is approximated by its *minimum bounding rectangle* (MBR)

1.**Filter step**: The **MBR** is tested against the query predicate.

2.**Refinement step**: The exact geometry of objects that pass the filter step is tested for qualification

### R-tree

**Build R-tree**:

Groups object MBRs to disk blocks hierarchically.

Each group of objects (a disk block) is a leaf of the tree

The MBRs of the leaf nodes are grouped to form nodes at the next level

Grouping is recursively applied at each level until a single group (the root) is formed

**Properties**:

leaf node entry : < MBR, obj-id >.    non-leaf node entry: < MBR , ptr >

Parameters(except Root) :  M, m , m<=M/2, usually m=0.4M

Root has at least 2 children, all leaves are in the same level,  1 node --> 1 block

**R*-tree**

R-tree and R*-tree differ only in the insertion algorithm.The improved insertion algorithm aims at constructing a tree of high quality.

**Good tree**:
*nodes with small MBRs*, *nodes with small overlaps*, *nodes that look like square*, *nodes as full as possible*

**Optimization Criteria**:
Minimize the area covered by an index rectangle
Minimize overlap between node MBRs
Minimize the margins of node MBRs 
Optimize the storage utilization (Nodes in tree should be filled as much as possible, Minimizes tree height and potentially decreases dead space)

**Insertion heuristics**
MBR is ***enlarged the least*** after insertion (n is non-leaf)
MBR enlargement will cause the ***minimum overlap*** with other entries of the same node (n is leaf) 
break any ties by choosing ***MBR with the minimum area***

**Node spliting**
If a node overflows we need to split it
Issue: distribute (fast!) a set of rectangles into two nodes such that the areas, overlap, and margins are minimized.

1.determine the split axisp

```java
For each axis (i.e. x and y axis)
Sum=0; 
sort entries by the lower value, then by upper value
for each sorting (e.g. lower value)
  for k=m to M+1-m 
    place first k entries in group A, and the remaining ones in group B
  	Sum = Sum + margin(A) + margin(B)

Choose axis with the minimum Sum
```

2.distribute entries along the axis
If there are multiple groupings with minimal overlap choose <A,B> such that area(A)+area(B) is minimized

**Insertion heuristics: forced reinsert**
1.When R*-tree node *n* overflows, instead of splitting n immediately, try to see if some entries in *n* could possibly fit better in another node
2.Find the 30% furthest entries from the center of the group
3.Re-insert them to the tree (not to be repeated if another overflow occurs)

Slightly more expensive, but better tree structure:
1.less overlap
2.more space is utilized (more full nodes)

**Bulk-loading R-trees**
Method1: iteratively insert rectangles into an initially empty tree
                   1.tree reorganization is slow
                   2.tree nodes are not as full as possible: more space occupied for the tree
Method2:(X-sorting) bulk-load the rectangles into the tree using some fast (sort or hash-based) processn
                   1.R-tree is built fast
				   2.good space utilization
Method3:(Hilbert sorting): use a space-filling curve to order the rectangles
 				  results in better structure than x-sorting

### Spatial Query

Spatial selection/range searching, nearest neighbor search, 

**Depth-first NN search using R-tree**
1.Start from the root and visit the node nearest to q
2.Continue recursively, until a leaf node nl is visited.
3.Find the NN of q in nl.
4.Continue visiting other nodes after backtracking as long there are nodes closer to q than the current NN.

<img src="https://github.com/toufunao/pic_repo/blob/main/2022-03-09/depth1_nn.png?raw=true" style="zoom:50%;" />

Advantages:
1.Large space can be pruned by avoiding visiting R-tree nodes and their sub-trees
2.Should order the entries of a node in increasing distance from q to maximize potential for a good NN found fast
3.Can be easily adapted for k-NN search (how?)
4.Requires at most one tree path to be currently in memory – good for small memory buffers

Disadvantages:
1.does not visit the least possible number of nodes
2.Also, not incremental 

**Best-first NN Search**

Use priority queue to organize and prioritize the next nodes to be visited.
<img src="https://github.com/toufunao/pic_repo/blob/main/2022-03-09/best1_nn.png?raw=true" style="zoom:50%;" />

Advantages:
1.we have visited fewer nodes compared to DF-NN algorithm
2.The algorithm can be adapted for incremental NN search
3.The algorithm can be used for k-NN search

Disadvantages:
The heap can grow very large until the algorithm terminates

**Spatial Join**

```shell
Input:
	two spatial relations R, S (e.g., R=cities, S=rivers)
	a spatial relationship θ (e.g., θ=intersects)

Output:
	{(r,s): rÎR, sÎS, r θ s is true}
	Example: find all pairs of cities and rivers that intersect
```











