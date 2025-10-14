"""
User Models

User management models including users, profiles, subscriptions, 
teams, and authentication-related data structures.
"""

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text, 
    ForeignKey, Enum, Float, BigInteger
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
from enum import Enum as PyEnum

from .base import EnhancedBaseModel

class UserRole(PyEnum):
    """User role enumeration"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM = "premium"
    STANDARD = "standard"
    TRIAL = "trial"

class UserStatus(PyEnum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    DELETED = "deleted"

class SubscriptionType(PyEnum):
    """Subscription type enumeration"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(PyEnum):
    """Subscription status enumeration"""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class User(EnhancedBaseModel):
    """User account model"""
    __tablename__ = "users"
    
    # Authentication fields
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # User information
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    display_name = Column(String(150), nullable=True)
    
    # Status and role
    role = Column(Enum(UserRole), default=UserRole.STANDARD, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING_VERIFICATION, nullable=False)
    
    # Authentication settings
    is_email_verified = Column(Boolean, default=False, nullable=False)
    is_2fa_enabled = Column(Boolean, default=False, nullable=False)
    
    # Verification tokens
    email_verification_token = Column(String(255), nullable=True)
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires = Column(DateTime(timezone=True), nullable=True)
    
    # Login tracking
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)  # IPv6 support
    login_count = Column(Integer, default=0, nullable=False)
    
    # Account preferences
    language = Column(String(10), default='en', nullable=False)
    timezone = Column(String(50), default='UTC', nullable=False)
    theme = Column(String(20), default='light', nullable=False)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    subscription = relationship("UserSubscription", back_populates="user", uselist=False, cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    team_memberships = relationship("TeamMember", back_populates="user", cascade="all, delete-orphan")
    activities = relationship("UserActivity", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.username
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium features"""
        return self.role in [UserRole.PREMIUM, UserRole.ADMIN, UserRole.MODERATOR]
    
    def can_create_projects(self, count: int = 1) -> bool:
        """Check if user can create more projects"""
        if self.role == UserRole.ADMIN:
            return True
        
        current_projects = len(self.projects)
        
        # Project limits by role
        limits = {
            UserRole.TRIAL: 1,
            UserRole.STANDARD: 5,
            UserRole.PREMIUM: 50,
            UserRole.MODERATOR: 100,
            UserRole.ADMIN: float('inf')
        }
        
        return current_projects + count <= limits.get(self.role, 0)

class UserProfile(EnhancedBaseModel):
    """Extended user profile information"""
    __tablename__ = "user_profiles"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, unique=True)
    
    # Personal information
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    website = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    
    # Social links
    social_links = Column(Text, nullable=True)  # JSON stored as text
    
    # Preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    marketing_emails = Column(Boolean, default=False, nullable=False)
    public_profile = Column(Boolean, default=False, nullable=False)
    
    # Statistics
    total_videos_created = Column(Integer, default=0, nullable=False)
    total_render_time = Column(Float, default=0.0, nullable=False)  # in hours
    total_storage_used = Column(BigInteger, default=0, nullable=False)  # in bytes
    
    # Relationship
    user = relationship("User", back_populates="profile")

class UserSubscription(EnhancedBaseModel):
    """User subscription management"""
    __tablename__ = "user_subscriptions"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False, unique=True)
    
    # Subscription details
    subscription_type = Column(Enum(SubscriptionType), default=SubscriptionType.FREE, nullable=False)
    status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE, nullable=False)
    
    # Billing information
    price = Column(Float, nullable=True)
    currency = Column(String(3), default='USD', nullable=False)
    billing_cycle = Column(String(20), nullable=True)  # monthly, yearly
    
    # Subscription period
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # External billing system
    external_subscription_id = Column(String(255), nullable=True)
    external_customer_id = Column(String(255), nullable=True)
    payment_method = Column(String(50), nullable=True)
    
    # Usage tracking
    monthly_render_minutes_used = Column(Integer, default=0, nullable=False)
    monthly_storage_used = Column(BigInteger, default=0, nullable=False)
    monthly_api_calls = Column(Integer, default=0, nullable=False)
    
    # Limits
    render_minutes_limit = Column(Integer, default=60, nullable=False)  # per month
    storage_limit = Column(BigInteger, default=1073741824, nullable=False)  # 1GB in bytes
    api_calls_limit = Column(Integer, default=1000, nullable=False)  # per month
    
    # Relationship
    user = relationship("User", back_populates="subscription")
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active"""
        return (self.status == SubscriptionStatus.ACTIVE and 
                (self.expires_at is None or self.expires_at > datetime.now(timezone.utc)))
    
    @property
    def days_remaining(self) -> int:
        """Get days remaining in subscription"""
        if not self.expires_at:
            return 999999  # Unlimited
        
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    def can_render(self, minutes: int) -> bool:
        """Check if user can render for given minutes"""
        return self.monthly_render_minutes_used + minutes <= self.render_minutes_limit
    
    def can_use_storage(self, bytes_needed: int) -> bool:
        """Check if user can use additional storage"""
        return self.monthly_storage_used + bytes_needed <= self.storage_limit
    
    def can_make_api_calls(self, calls: int) -> bool:
        """Check if user can make API calls"""
        return self.monthly_api_calls + calls <= self.api_calls_limit

class Team(EnhancedBaseModel):
    """Team management model"""
    __tablename__ = "teams"
    
    # Team information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    
    # Team settings
    is_public = Column(Boolean, default=False, nullable=False)
    max_members = Column(Integer, default=10, nullable=False)
    
    # Owner
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Relationships
    owner = relationship("User")
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="team", cascade="all, delete-orphan")
    
    @property
    def member_count(self) -> int:
        """Get team member count"""
        return len(self.members)
    
    def can_add_members(self, count: int = 1) -> bool:
        """Check if team can add more members"""
        return self.member_count + count <= self.max_members

class TeamMember(EnhancedBaseModel):
    """Team membership model"""
    __tablename__ = "team_members"
    
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id'), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Member role within team
    role = Column(String(50), default='member', nullable=False)  # owner, admin, member, viewer
    
    # Permissions
    can_edit_projects = Column(Boolean, default=True, nullable=False)
    can_delete_projects = Column(Boolean, default=False, nullable=False)
    can_manage_members = Column(Boolean, default=False, nullable=False)
    can_manage_settings = Column(Boolean, default=False, nullable=False)
    
    # Membership status
    status = Column(String(20), default='active', nullable=False)  # active, pending, suspended
    invited_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    joined_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")
    
    def __repr__(self):
        return f"<TeamMember(team_id={self.team_id}, user_id={self.user_id}, role={self.role})>"